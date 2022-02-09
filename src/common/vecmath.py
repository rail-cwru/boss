"""
Useful math functions.
"""

import numpy as np
import numba


def vec_choice(prob_matrix, items):
    """
    Choose N items with probabilities from prob_matrix with shape [N, I].
    :param prob_matrix: Shape [N, I] array with probabilities.
    :param items: Selection array of size [I, ...]
    :return: Chosen items [N, ...]
    """
    prob_matrix = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)
    s = prob_matrix.cumsum(axis=1)
    r = np.random.rand(prob_matrix.shape[0])[:, None]
    k = np.less(s, r).sum(axis=1)
    return items[k]


# @profile
def argmax_random_tiebreak(a: np.ndarray, axis=1) -> np.ndarray:
    """
    General argmax with random tiebreaking.
    """
    if a.ndim == 1:
        return np.random.choice(np.flatnonzero(a == a.max()))
    # Else more than one dimension
    if a.ndim > 2:
        # Pretend it's two dimensions
        arr2 = np.swapaxes(a, axis, 0)
        target_shape = arr2.shape[1:]
        arr2 = arr2.reshape(a.shape[axis], -1)
        axis = 0
    else:
        arr2 = a
    max_mask = (arr2 == arr2.max(axis=axis, keepdims=True))
    L = max_mask.sum(axis=axis)
    set_mask = np.zeros(L.sum(), dtype=bool)
    set_mask[__random_num_per_grp_cumsum(L)] = True
    if axis == 0:
        max_mask.T[max_mask.T] = set_mask
    else:
        max_mask[max_mask] = set_mask
    ret = max_mask.argmax(axis=axis)
    if a.ndim > 2:
        return ret.reshape(target_shape)
    else:
        return ret


# @profile
def nd_gather(arr: np.ndarray, nd_index: np.ndarray, axis=-1) -> np.ndarray:
    """
    Effectively nd_gather.
    """
    return np.squeeze(arr[tuple(np.split(nd_index, nd_index.shape[axis], axis=axis))], axis=axis)


# @profile
def __random_num_per_grp_cumsum(L):
    # For each element in L pick a random number within range specified by it
    # The final output would be a cumsumed one for use with indexing, etc.
    r1 = np.random.rand(np.sum(L)) + np.repeat(np.arange(len(L)), L)
    offset = np.r_[0, np.cumsum(L[:-1])]
    return r1.argsort()[offset]


@numba.njit()
def _bcj2s1(out, a1, a2, y1, y2):
    for i in range(y1):
        out[i::y1, 0] = a1
    for i in range(y2):
        out[i*y2:(i+1)*y2, 1] = a2


@numba.njit()
def _bcj3s2(out, a1, a2, y1, y2):
    for i in range(y1):
        out[:, i::y1, 0] = a1
    for i in range(y2):
        out[:, i*y2:(i+1)*y2, 1] = a2


@numba.njit('f4[:,:],i4[:,:],i4[:],f4[:]')
def _maxtiebreak_2d_ax0_seq(arr, indices, argmax, rowmax):
    # Mutating operation that performs argmax-tiebreak and stores to argmax and rowmax.
    # Doing it in one loop isn't actually speedup. For some reason.
    # Sequential ver. Less overhead makes it faster on smaller arrays.
    rows, cols = arr.shape
    for j in range(cols):
        rowmax_num = np.max(arr[:, j])
        rowmax[j] = rowmax_num
        count = 0
        for i in range(rows):
            if arr[i, j] == rowmax_num:
                indices[count, j] = i
                count += 1
        argmax[j] = indices[np.random.randint(count), j]


@numba.njit('f4[:,:],i4[:,:],i4[:],f4[:]', parallel=True)
def _maxtiebreak_2d_ax0_par(arr, indices, argmax, rowmax):
    # Mutating operation that performs argmax-tiebreak and stores to argmax and rowmax.
    # Parallel ver. Speeds up on larger arrays.
    rows, cols = arr.shape
    for j in numba.prange(cols):
        rowmax_num = np.max(arr[:, j])
        rowmax[j] = rowmax_num
        count = 0
        for i in range(rows):
            if np.float32(arr[i, j]) == np.float32(rowmax_num):
                indices[count, j] = i
                count += 1
        argmax[j] = indices[np.random.randint(count), j]


class _CachedVecmath(object):
    """
    Performs math with cached array buffer.
    Use only when the output array is immediately used and thrown away, as data may be modified after return.
    """

    def __init__(self):
        self.cache = {}
        self.argmax_cache = {}

    def argmax_tiebreak(self, a: np.ndarray, axis: int):
        """
        Argmax with random tiebreak - wrapper function.
        Returns [argmax, rowmax].
        """
        if a.ndim == 1:
            return np.random.choice(np.flatnonzero(a == a.max())), a.max(axis=axis)
        elif a.ndim == 2:
            a = a if axis == 0 else a.T
            return self._maxtiebreak_2d_ax0(a)
        else:
            a = np.swapaxes(a, axis, 0)
            reduce_size, target_shape = a.shape[0], a.shape[1:]
            a = a.reshape(reduce_size, -1)
            argmax, rowmax = self._maxtiebreak_2d_ax0(a)
            return argmax.reshape(target_shape), rowmax.reshape(target_shape)

    def _maxtiebreak_2d_ax0(self, arr):
        assert arr.dtype == np.float32
        if arr.shape not in self.argmax_cache:
            indices = np.zeros(arr.shape, dtype=np.int32)
            argmax = np.zeros(arr.shape[1], dtype=np.int32)
            rowmax = np.zeros(arr.shape[1], dtype=np.float32)
            self.argmax_cache[arr.shape] = (indices, argmax, rowmax)
        else:
            indices, argmax, rowmax = self.argmax_cache[arr.shape]
        if arr.size <= 16000:
            _maxtiebreak_2d_ax0_seq(arr, indices, argmax, rowmax)
        else:
            _maxtiebreak_2d_ax0_par(arr, indices, argmax, rowmax)
        return np.copy(argmax), np.copy(rowmax)

    def broadcast_joint_3d_stack2(self, a1, a2):
        # assert z1 == z2 and x1 == x2
        z1, y1, x1 = a1.shape
        z2, y2, x2 = a2.shape
        key = (z1, y1*y2, 2, x1)
        if key not in self.cache:
            out = np.zeros(key, dtype=np.float32)
            self.cache[key] = out
        else:
            out = self.cache[key]
        _bcj3s2(out, a1, a2, y1, y2)
        return out

    def broadcast_joint_2d_stack1(self, a1, a2):
        y1, x1 = a1.shape
        y2, x2 = a2.shape
        key = (y1 * y2, 2, x1)
        if key not in self.cache:
            out = np.zeros(key, dtype=np.float32)
            self.cache[key] = out
        else:
            out = self.cache[key]
        _bcj2s1(out, a1, a2, y1, y2)
        return out

    def retrieve(self, shape, slot=None):
        # Try to directly write into functions for when metaprogramming and macros are possible in the future
        # For now we can also use this from the outside.
        key = shape if slot is None else (str(slot),) + shape
        if key not in self.cache:
            out = np.zeros(shape, dtype=np.float32)
            self.cache[key] = out
        else:
            out = self.cache[key]
        return out

    def clear(self):
        self.cache.clear()


# Singletons
CachedOps = _CachedVecmath()


def norm_l1_ball(radius: (int, float), ndims: int) -> np.ndarray:
    """
    Construct a list of vectors in a closed ball in `ndims` dimensions of `radius`.
    :param radius: Radius of ball
    :param ndims: Dimension of ball
    :return: List (as ndarray) of vectors in and on closed ball.

    It is possible to calculate this in efficient way without any "extra" computation,
        but it is mathematically frustrating to do an ordering for {vec : ||vec||_1 <= radius, vec in R^ndims}.

    There is absolutely, certainly a better way to calculate this.
    """
    assert ndims <= 8, 'L1-ball not supported for dim > 8 due to memory concerns.'
    x = np.arange(-radius, radius + 1)
    stacked = np.stack(np.meshgrid(*[x] * ndims), axis=-1)
    return stacked[np.abs(stacked).sum(axis=-1) <= radius]
