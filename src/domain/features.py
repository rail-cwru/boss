"""
For atoms of information regarding observation and feature encoding.

Check feature type via the feature_type Enum.
    Do not use isinstance, as "copy" will produce base Features, though otherwise identical.

    (Maybe the various types of features should be functions which happen to construct feature appropriately?)
"""
from enum import Enum
from typing import Any, List, Union, Optional
import numpy as np

# ENUMS
from domain.items import DomainItem


# TODO - Feature range checking. Make sure that values written to features do not exceed their described range.
class FeatureType(Enum):
    NONE = 0
    COORDINATE = 1
    BINARY = 2
    ONE_HOT = 3
    REAL = 4
    DISCRETE = 5
    VECTOR = 6
    NORMALIZED_REAL = 7

class Feature(DomainItem):
    """
    DomainItems used for observation / environment state representation.
    """

    def __init__(self,
                 name: str,
                 shape: List[int],
                 dtype: str,
                 drange: slice=slice(None),
                 prefix: Any=None,
                 feature_type=FeatureType.NONE):
        # TODO drange is not meaningful and should be refactored / removed.
        self.feature_type = feature_type
        super(Feature, self).__init__(name, shape, dtype, drange, prefix)

    def num_values(self):
        raise TypeError("An arbitrary feature does not guarantee that it is restricted to "
                        "some number of distinct values.")


class CoordinateFeature(Feature):

    def __init__(self,
                 name: str,
                 lower: List[Union[float, int]],
                 upper: List[Union[float, int]],
                 is_discrete: bool,
                 sparse_values: Optional[List[List[Union[float, int]]]]=None,
                 prefix: Any = None,):
        """
        CoordinateFeature represents features which are a spatial coordinate bounded by two other coordinates.
            i.e. in 2D, CoordinateFeature represents a point lying within a square or rectangle
                 in 3D, CoordinateFeature represents a point lying within a cuboidal region
                 etc.

        The upper bound is non-inclusive.

        :param name: Name of feature
        :param lower: The inclusive lower bound for the region.
        :param upper: The upper bound for the region. Inclusive if continuous. Exclusive if discrete.
        :param is_discrete: If the coordinate is int-valued or float-valued.
        :param sparse_values: If specified, restricts the coordinate feature to only taking on the values in the list.
        :param prefix: A named prefix for the feature.
        """
        dtype = 'int' if is_discrete else 'float'
        self.ndim = len(lower)
        assert self.ndim > 0, "Coordinate feature must have at least one dimension."
        assert self.ndim == len(upper),\
            "The coordinate feature [{}] was generated with inconsistent bound dimensions [{}] and [{}]"\
            .format(name, self.ndim, len(upper))
        d_min, d_max = min(lower), max(upper)  # TODO remove on drange refactor
        self.lower = lower
        self.upper = upper
        self.region_dims = [hi - lo for lo, hi in zip(lower, upper)]
        self.is_infinite = d_max == float('inf') or d_min == float('-inf')
        if sparse_values:
            self.__num_values = len(sparse_values)
            for value in sparse_values:
                assert len(value) == self.ndim
                if is_discrete:
                    assert all([isinstance(v, int) for v in value]), \
                        "Sparse values must be int for discrete CoordinateFeature."
                    for v, lo, hi in zip(value, lower, upper):
                        assert lo <= v < hi, "Sparse value was outside range"
                else:
                    for v, lo, hi in zip(value, lower, upper):
                        assert lo <= v <= hi, "Sparse value was outside range"
            self.sparse_values = sparse_values
        else:
            self.sparse_values = []
            if (not self.is_infinite) and is_discrete:
                self.__num_values = int(np.prod(self.region_dims))
            else:
                self.__num_values = float('inf')
        # Bound checking
        for dim in self.region_dims:
            if is_discrete:
                assert dim > 0, "Illegal discrete CoordinateFeature: Lower bound was greater or equal than upper bound."
            else:
                assert dim >= 0, "Illegal continuous CoordinateFeature: Lower bound was greater than upper bound."
        super().__init__(name=name,
                         shape=[self.ndim],
                         dtype=dtype,
                         drange=slice(d_min, d_max),
                         prefix=prefix,
                         feature_type=FeatureType.COORDINATE)

    def num_values(self):
        return self.__num_values


class VectorFeature(Feature):

    def __init__(self,
                 name: str,
                 ndim: int,
                 min: float,
                 max: float,
                 prefix: Any = None):
        """
        A feature representing a real-valued vector.
        If there are bounds, they are defined over all values in the vector.

        :param name: Name of feature
        :param ndim: Number of dimensions
        :param min: Minimum value for any entry in the vector.
        :param max: Maximum value for any entry in the vector. Inclusive if float-valued, otherwise exclusive.
        """
        super().__init__(name=name,
                         shape=[ndim],
                         dtype='float',
                         drange=slice(min, max),
                         prefix=prefix,
                         feature_type=FeatureType.VECTOR)

    def num_values(self):
        return float('inf')


class BinaryFeature(Feature):

    def __init__(self,
                 name: str,
                 prefix: Any = None):
        """
        A feature representing a single binary value.
        :param name: Name of feature
        """
        super().__init__(name=name,
                         shape=[1],
                         dtype='int',
                         drange=slice(0, 2),
                         prefix=prefix,
                         feature_type=FeatureType.BINARY)

    def num_values(self):
        return 2


class DiscreteFeature(Feature):

    # FIXME removed 1 from starts_from+size+1, since this adds an extra value
    def __init__(self, name: str, size: int, starts_from: int = 0, prefix: Any = None):
        """
        A feature represented by a discrete value starting from 0 (or optionally, another number)

        :param name: Name of feature
        :param size: Size of feature (number of discrete options)
        :param starts_from: The first number used to index the feature. 0 by default.
        :param prefix: Prefix to name (optional)
        """
        super().__init__(name=name,
                         shape=[1],
                         dtype='int',
                         drange=slice(starts_from, starts_from + size),
                         prefix=prefix,
                         feature_type=FeatureType.DISCRETE)

    def num_values(self):
        return self.range.stop - self.range.start


class PlusMinusFeature(Feature):

    def __init__(self, name: str, prefix: Any = None):
        """
        A special form of DiscreteFeature which covers the values [-1, 0, 1].
        :param name: Name of feature
        :param prefix: Prefix to name (optional)
        """
        super().__init__(name=name,
                         shape=[1],
                         dtype='int',
                         drange=slice(-1, 2),
                         prefix=prefix,
                         feature_type=FeatureType.DISCRETE)

    def num_values(self):
        return 3


class RealFeature(Feature):

    def __init__(self, name: str, min: Optional[float]=None, max: Optional[float]=None, prefix: Any = None):
        """
        A feature represented by a single real value.

        :param name: Name of feature
        :param min: Min value - otherwise unbounded
        :param max: Max value - otherwise unbounded
        :param prefix: Prefix to name (optional)
        """
        super().__init__(name=name,
                         shape=[1],
                         dtype='float',
                         drange=slice(min, max),
                         prefix=prefix,
                         feature_type=FeatureType.REAL)

    def num_values(self):
        return float('inf')


class NormalizedRealFeature(Feature):

    def __init__(self, name: str, prefix: Any = None):
        """
        A feature represented by a single real value.

        :param name: Name of feature
        :param min: Min value
        :param max: Max value
        :param prefix: Prefix to name (optional)
        """
        super().__init__(name=name,
                         shape=[1],
                         dtype='float',
                         drange=slice(0, 1),
                         prefix=prefix,
                         feature_type=FeatureType.NORMALIZED_REAL)

    def num_values(self):
        return float('inf')


class OneHotFeature(Feature):

    def __init__(self,
                 name: str,
                 num_categories_or_list: (int, list),
                 prefix: Any = None):
        """
        A feature representing an exclusive categorical type.
        For example, if something is a CAT or a DOG...
        Types of information which is generally well-represented as a one-hot vector.

        :param name: Name of feature
        :param num_categories_or_list: Number or list of categories
        """
        if isinstance(num_categories_or_list, int):
            self.num_categories = num_categories_or_list
            self.categories = list(np.r_[:num_categories_or_list])
        elif isinstance(num_categories_or_list, list):
            self.num_categories = len(num_categories_or_list)
            self.categories = num_categories_or_list
        else:
            raise ValueError('[{}] not int or list'.format(num_categories_or_list))

        super().__init__(name=name,
                         shape=[self.num_categories],
                         dtype='int',
                         drange=slice(0, 2),
                         prefix=prefix,
                         feature_type=FeatureType.ONE_HOT)

    def num_values(self):
        return self.num_categories
