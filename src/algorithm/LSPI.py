from typing import List, Tuple
import torch
from torch.nn import CrossEntropyLoss

import numpy as np
from scipy.special import logsumexp
from common.properties import Properties
from config import Config, ConfigItemDesc, checks, ConfigDesc
from common.trajectory import Trajectory
from policy.function_approximator.pytorch_fa import AbstractPyTorchFA
from policy.function_approximator.ACFunctionApproximator import ACFunctionApproximator
from policy.function_approximator.SimpleLinear import SimpleLinear
from . import Algorithm
from policy import Policy
from model import Model
from .util import MemoryReplay, calc_discounted_rewards
from policy.function_approximator.basis_function.ExactBasis import ExactBasis
from domain.conversion import FeatureConversions
import scipy
from scipy.sparse.linalg import spsolve
import time
import torch
from numba import jit, njit
# from pympler import asizeof
# import sys
# import tensorflow as tf
# from tensorflow.python.client import device_lib
# CUDA_VISIBLE_DEVICES = 0

@jit(nopython=True)
def solve_numpy_njit(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    parameters = np.linalg.solve(a, b)
    return parameters


class LSPI(Algorithm):
    """
    This is an implementation of the least-squares policy iteration (LSPI) algorithm. This is an offline learning
    algorithm that receives all of the samples and solves for the optimal policy using matrix operations. A basis
    function is required to convert the state to an array, which allows LSPI to be used with linear state approximators.
    To replicate a tabular representation, use the Exact Basis function. This framework currently supports 2 forms of
    derived samples, inhibited and abstract samples, taken from a task hierarchy

    @author Eric Miller
    @contact ericm@case.edu

    The derived samples are discussed by Schwab and Ray in "Offline Reinforcement Learning with Task Hierarchies"

    The LSPI algorithm was orignally proposed by Lagoudakis and Parr
    More information about the LSPI and Matlab source code is available here:
    https://www2.cs.duke.edu/research/AI/LSPI/
     """


    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return Algorithm.get_class_config() + []

    @classmethod
    def properties_helper(cls):
        return Properties(
            use_function_approximator=True,
            pytorch=True)

    def __init__(self, config: Config):
        Algorithm.__init__(self, config)
        self.memory = MemoryReplay(self.config.memory_size, self.config.batch_size, False, self.config.update_interval)


    # This code was taken from the Reinforce algorithm, and truthfully I am not sure if it is needed
    def compile_policy(self, policy: Policy):
        # TODO we dont need nearly this much (?)
        pytorch_cross_entropy = CrossEntropyLoss(reduction='none')

        def numpy_cross_entropy(probs, actions):
            lse = logsumexp(probs, axis=1)
            ce = -probs[range(actions.shape[0]), actions] + lse
            return ce

        def numpy_loss(data: np.array):
            raise Exception('Numpy version of Reinforce is not complete.')

            # Gather states, actions and discounted rewards
            num_episodes = len(data)
            batch_states = np.array([])
            batch_actions = np.array([])
            batch_discounted_rewards = np.array([])
            for i in range(num_episodes):
                states, actions, rewards, _, _, _ = data[i]

                batch_states = np.concatenate((batch_states, states), axis=0)
                batch_actions = np.concatenate((batch_actions, actions), axis=0)
                discounted_rewards = calc_discounted_rewards(self.discount_factor, rewards)
                batch_discounted_rewards = np.concatenate((batch_discounted_rewards, discounted_rewards), axis=0)

            # Get function approximator values
            fa_values = policy.function_approximator.eval(batch_states)

            # Get probabilities from sampler
            probs = np.zeros(fa_values.shape)
            for i in range(fa_values.shape[0]):
                _, probs[i,:] = policy.sampler.raw_sample(fa_values[i,:])

            # Use cross entropy to control numerical errors
            selected_log_probs = numpy_cross_entropy(probs, batch_actions)

            # Finish loss calc
            advantage = selected_log_probs * batch_discounted_rewards
            sum_advantage = np.sum(advantage)
            return sum_advantage

        def pytorch_loss(data: np.array):
            # Gather states, actions and discounted rewards
            num_episodes = len(data)
            batch_states = None
            batch_actions = None
            batch_discounted_rewards = None
            for i in range(num_episodes):
                states, actions, rewards, _, _, _ = data[i]

                if policy.domain_act.is_compound:
                    actions = self.translate_compound_action(policy.domain_act, actions)
                actions = np.reshape(actions, (actions.shape[0]))
                discounted_rewards = calc_discounted_rewards(self.discount_factor, rewards, True)

                if i == 0:
                    batch_states = states
                    batch_actions = actions
                    batch_discounted_rewards = discounted_rewards
                else:
                    batch_states = np.concatenate((batch_states, states), axis=0)
                    batch_actions = np.concatenate((batch_actions, actions), axis=0)
                    batch_discounted_rewards = np.concatenate((batch_discounted_rewards, discounted_rewards), axis=0)

            # Convert to tensors
            x_var = torch.tensor(batch_states, dtype=torch.float, device=policy.function_approximator.device)
            a_var = torch.tensor(batch_actions, dtype=torch.long, device=policy.function_approximator.device)
            dr_var = torch.tensor(batch_discounted_rewards, dtype=torch.float, device=policy.function_approximator.device)

            # Predict probability from policy
            fa_values = policy.function_approximator.eval_tensor(x_var)

            # Get probabilities from sampler
            prob = policy.sampler.sample_tensor(fa_values)

            # Use cross entropy to control numerical errors
            selected_log_probs = pytorch_cross_entropy(prob, a_var)

            # Finish loss calc
            advantage = torch.mul(selected_log_probs, dr_var)
            sum_advantage = torch.sum(advantage)
            return sum_advantage

        # Check to make sure not actor critic
        fa = policy.function_approximator
        assert not isinstance(fa, ACFunctionApproximator)

        # Compile policy
        if isinstance(fa, AbstractPyTorchFA):
            policy.compile(pytorch_loss, 1.0)
        else:
            policy.compile(numpy_loss, 1.0)

    def update(self, policy: Policy, model: Model, trajectory: Trajectory):
        raise TypeError('LSPI designed for updating in this manner, consider using Reinforce for offline learning')

    def learn(self, samples, policy_groups, epsilon=1e-4, max_iterations=100, single_list=False, weights=None, sampler=None):
        """
        The entry point for the LSPI algorithm, connects to the solvers to find the optimal policy given the samples

        :param samples: A list a np.ndarrays, each with the following structure:
            [state, action, reward, next state]
            If there is no next state, None should be used
        :param policy_groups: A list of policy groups for agents in the environment. Note only single-agent learning is
        currently supported
        :param epsilon: The max error rate for convergence
        :param max_iterations: Limit of iterations
        :return: The policy is update via weights, so nothing is returned
        """
        if epsilon <= 0:
            raise ValueError('epsilon must be > 0: %g' % epsilon)
        if max_iterations <= 0:
            raise ValueError('max_iterations must be > 0: %d' % max_iterations)

        for pg in policy_groups:
            curr_policy = pg.policy
            distance = float('inf')
            iteration = 0
            while distance > epsilon and iteration < max_iterations:
                iteration += 1

                # The exact solver is optimized for the exact basis function, which tends to run quite slowly
                if isinstance(curr_policy.function_approximator.basis_function, ExactBasis):
                    if single_list:
                        new_weights = self.lstdq_solver_exact_single(samples, curr_policy, weights=weights)
                    else:
                        new_weights = self.lstdq_solver_exact(samples, curr_policy, weights=weights)
                else:
                    new_weights = self.lstdq_solver(samples, curr_policy)

                distance = np.linalg.norm(new_weights-curr_policy.function_approximator.get_weights())
                curr_policy.function_approximator.set_weights(new_weights)

            if distance > epsilon:
                if sampler:
                    print('Hit max iterations, may not be fully converged', distance, sampler)
                else:
                    print('Hit max iterations, may not be fully converged', distance)


    def lstdq_solver(self, samples, policy, precondition_value=.1):
        """
        The normal LSTQ solver, suitable for any basis function

        :param samples: The list of samples. Each sample is a np.ndarray with structure:
            [state, action, reward, next state]
            If there is no next state, None should be used
        :param policy: Current policy to improve
        :param precondition_value: value to initialize a_mat to
        :return: The newly updated weights for the policy
        """

        basis = policy.function_approximator.basis_function
        agent_id = 0
        k = basis.size()
        a_mat = np.zeros((k, k))
        np.fill_diagonal(a_mat, precondition_value)
        b_vec = np.zeros((k, 1))
        for episode in samples:
            max_t = episode.time
            episode = episode.get_agent_trajectory(agent_id)
            actions = episode.actions
            rewards = episode.rewards
            observations = episode.observations
            for index in range(max_t):
                start = time.time()
                mapped_obs = np.asarray(self._map_table_indices(observations[index], policy))

                phi_sa = basis.evaluate(mapped_obs, actions[index]).reshape((-1, 1))
                print(time.time() - start, 'Time Elapsed for Step 1')
                start = time.time()
                if index == max_t - 1:
                    # Ended episode
                    phi_sprime = np.zeros((k, 1), dtype=np.float32)
                else:
                    next_obs = self._map_table_indices(observations[index + 1], policy)

                    best_action = policy.function_approximator.best_action(next_obs)
                    phi_sprime = policy.function_approximator.basis_function.evaluate(next_obs, best_action).reshape((-1, 1))

                v1 = np.asarray((phi_sa - self.discount_factor * phi_sprime).T, dtype=np.float32)
                v2 = np.asarray(phi_sa, dtype=np.float32)

                p = v2.dot(v1)
                a_mat += p
                b_vec += v2 * rewards[index]

        a_rank = np.linalg.matrix_rank(a_mat)
        if a_rank == k:
            w = scipy.linalg.solve(a_mat, b_vec)
        else:
            print('A matrix is not full rank. %d < %d', a_rank, k)
            w = scipy.linalg.lstsq(a_mat, b_vec)[0]
        return w.reshape((-1, ))

    def lstdq_solver_exact(self, samples, policy, precondition_value=.1, weights=None):
        """
        A LSTDQ solver optimized for the Exact basis function
        Only for use with exact basis function

        :param samples: The list of samples. Each sample is a np.ndarray with structure:
            [state, action, reward, next state]
            If there is no next state, None should be used
        :param policy: Current policy to improve
        :param precondition_value: value to initialize a_mat to
        :return: The newly updated weights for the policy
        """
        basis = policy.function_approximator.basis_function
        if not isinstance(basis, ExactBasis):
            raise TypeError("Only Exact Basis Function Will Work")

        k = basis.size()
        try:
            a_mat = np.zeros((k, k), dtype=np.float32)
            np.fill_diagonal(a_mat, precondition_value)
        except:
            raise ValueError("Too many state variables to allocate array on this computer")

        b_vec = np.zeros((k, 1))

        for episode in samples:
            for sample in episode:
                mapped_obs = np.asarray(self._map_table_indices(sample[0], policy))

                phi_sa, sa_ind = basis.evaluate2(mapped_obs,sample[1])
                phi_sa = phi_sa.reshape((-1, 1))

                if sample[-1] is None:
                    # Ended episode
                    sprime_ind = -1
                else:
                    next_obs = self._map_table_indices(sample[-1], policy)
                    best_action = policy.function_approximator.best_action(next_obs)
                    phi_sprime, sprime_ind = policy.function_approximator.basis_function.evaluate2(next_obs,
                                                                                                   best_action)
                a_mat[sa_ind, sa_ind] += 1
                if sprime_ind >= 0:
                    a_mat[sa_ind, sprime_ind] -= self.discount_factor

                v2 = np.asarray(phi_sa, dtype=np.float32)
                # Add sample reward
                b_vec += v2 * sample[2]
        rank = False
        a_mat_csr = scipy.sparse.csr_matrix(a_mat)
        # Ranking is quite slow, usually can safely (?) assume the rank is fine
        if rank:
            a_rank = np.linalg.matrix_rank(a_mat)
            if a_rank == k:
                w = spsolve(a_mat_csr, b_vec)
            else:
                print('A matrix is not full rank. %d < %d', a_rank, k)
                w = scipy.linalg.lstsq(a_mat, b_vec)[0]
        else:

            a_mat_csr = scipy.sparse.csr_matrix(a_mat)
            w = spsolve(a_mat_csr, b_vec)
        return w.reshape((-1,))

    def lstdq_solver_exact_single(self, samples, policy, precondition_value=.1, weights=None):
        """
        A LSTDQ solver optimized for the Exact basis function
        Only for use with exact basis function

        :param samples: The list of samples. Each sample is a np.ndarray with structure:
            [state, action, reward, next state]
            If there is no next state, None should be used
        :param policy: Current policy to improve
        :param precondition_value: value to initialize a_mat to
        :return: The newly updated weights for the policy
        """
        basis = policy.function_approximator.basis_function
        if not isinstance(basis, ExactBasis):
            raise TypeError("Only Exact Basis Function Will Work")

        k = basis.size()
        try:
            a_mat = scipy.sparse.lil_matrix((k, k), dtype=float)
            a_mat.setdiag([precondition_value] * k)
            b_vec = scipy.sparse.lil_matrix((k, 1), dtype=float)

        except:
            raise ValueError("Too many state variables to allocate array on this computer")

        for ind, sample in enumerate(samples):
            sa_ind = sample[0]
            if sample[-1] == []:
                # Ended episode
                sprime_ind = -1
            else:
                sa_inds = sample[-1]
                best_action, sprime_ind = policy.function_approximator.best_action2(sa_inds)
            a_mat[sa_ind, sa_ind] += 1

            if sprime_ind >= 0 and weights:
                a_mat[sa_ind, sprime_ind] -= self.discount_factor * weights[ind]
            elif sprime_ind >= 0:
                a_mat[sa_ind, sprime_ind] -= self.discount_factor

            if weights:
                b_vec[sa_ind, 0] += sample[-2] * weights[ind]
            else:
                b_vec[sa_ind, 0] += sample[-2]

        rank = False
        # Ranking is quite slow, usually can safely (?) assume the rank is fine
        if rank:
            a_rank = np.linalg.matrix_rank(a_mat)
            if a_rank == k:
                w = spsolve(a_mat, b_vec)
            else:
                print('A matrix is not full rank. %d < %d', a_rank, k)
                w = scipy.linalg.lstsq(a_mat, b_vec)[0]
        else:
            w = spsolve(a_mat.tocsr(), b_vec.tocsr())
        return w.reshape((-1,))

    def verify_sparse_solver(self, a_mat, b_vec):
        """
        A method to check that the scipy sparse matrix solver is returning the correct answer
        :return: True if the scipy sparse-solver works
        """
        a_mat_csr = scipy.sparse.csr_matrix(a_mat)
        w = spsolve(a_mat_csr, b_vec)
        w1 = np.linalg.solve(a_mat, b_vec)

        s = True
        for index, i in enumerate(w):
            if abs(i - w1[index]) > 1e-4:
                print(i, w1[index])
                s = False
        return s

    def lstdq_be_solver(self, samples, policy, precondition_value=.1):
        """
        A solver adapted from the original LSPI Matlab Code, not currently functional
        :param samples: Typically the list of samples
        :param policy: Current policy to improve
        :return: new policy with updated weights
        """
        basis = policy.function_approximator.basis_function
        agent_id = 0
        k = basis.size()
        a_mat = np.zeros((k, k))
        np.fill_diagonal(a_mat,precondition_value)
        b_vec = np.zeros((k, 1))
        start = time.time()
        for episode in samples:
            max_t = episode.time
            episode = episode.get_agent_trajectory(agent_id)
            actions = episode.actions
            rewards = episode.rewards
            observations = episode.observations
            for index in range(max_t):
                mapped_obs = np.asarray(self._map_table_indices(observations[index], policy))
                mapped_obs[-1] += 1

                phi_sa = basis.evaluate(mapped_obs, actions[index]).reshape((-1, 1))
                if index == max_t - 1:
                    # Ended episode
                    phi_sprime = np.zeros((k, 1), dtype=np.float32)
                else:
                    next_obs = self._map_table_indices(observations[index + 1], policy)
                    next_obs[-1] += 1
                    best_action = policy.function_approximator.best_action(next_obs)
                    phi_sprime = policy.function_approximator.basis_function.evaluate(next_obs, best_action).reshape((-1, 1))

                v3 = np.asarray(phi_sa - self.discount_factor * phi_sprime, dtype=np.float32)
                p = v3.dot(v3.T)
                a_mat += p
                b_vec += v3 * rewards[index]

        a_rank = np.linalg.matrix_rank(a_mat)
        if a_rank == k:
            w = scipy.linalg.solve(a_mat, b_vec)
        else:
            print('A matrix is not full rank. %d < %d', a_rank, k)
            w = scipy.linalg.lstsq(a_mat, b_vec)[0]
        print(time.time() - start, 'Time Elapsed for 1 iteration')
        return w.reshape((-1, ))

    def lstdq_solver_fast(self, samples, policy, precondition_value=.1, first_time=False):
        """
        An optimized LSTDQ for any basis function, not yet functional

        :param samples: Typically the list of samples
        :param policy: Current policy to improve
        :return: new policy with updated weights
        """
        basis = policy.function_approximator.basis_function
        agent_id = 0
        k = basis.size()
        a_mat = np.zeros((k, k))
        np.fill_diagonal(a_mat,precondition_value)
        b_vec = np.zeros((k, 1))

        if first_time:
            self.how_many = 0
            for episode in samples:
                self.how_many += episode.time

            self.phihat = np.ndarray(shape=(self.how_many, k))
            self.rhat = np.ndarray(shape=(self.how_many, 1))
            running_tot = 0
            for i, episode in enumerate(samples):
                max_t = episode.time
                episode = episode.get_agent_trajectory(agent_id)
                actions = episode.actions
                rewards = episode.rewards
                observations = episode.observations

                for index in range(max_t):
                    mapped_obs = np.asarray(self._map_table_indices(observations[index], policy))
                    mapped_obs[-1] += 1
                    phi_sa = basis.evaluate(mapped_obs, actions[index]).reshape((-1, 1))
                    self.phihat[running_tot] = (phi_sa).T
                    self.rhat[running_tot] = rewards[index]
                    running_tot += 1

        pi_phi_hat = np.ndarray(shape=(self.how_many, policy.function_approximator.basis_function.size()))
        running_tot = 0
        for i, episode in enumerate(samples):
            max_t = episode.time
            episode = episode.get_agent_trajectory(agent_id)
            observations = episode.observations
            for index in range(max_t):
                if index == max_t - 1:
                    # Ended episode
                    phi_sprime = np.zeros((k, 1), dtype=np.float32)
                else:
                    next_obs = self._map_table_indices(observations[index + 1], policy)
                    next_obs[-1] += 1
                    best_action = policy.function_approximator.best_action(next_obs)
                    phi_sprime = policy.function_approximator.basis_function.evaluate(next_obs, best_action).reshape((-1, 1))
                    pi_phi_hat[running_tot] = phi_sprime.T
                running_tot += 1

        a_mat = self.phihat.T.dot((self.phihat - self.discount_factor* pi_phi_hat))
        b_vec = self.phihat.T.dot(self.rhat)

        print(b_vec.shape, a_mat.shape)

        a_rank = np.linalg.matrix_rank(a_mat)
        if a_rank == k:
            w = scipy.linalg.solve(a_mat, b_vec)
        else:
            print('A matrix is not full rank. %d < %d', a_rank, k)
            w = scipy.linalg.lstsq(a_mat, b_vec)[0]

        print(w.shape)
        return w.reshape((-1, ))

    def _map_table_indices(self, states: np.ndarray, policy: Policy) -> Tuple[int, ...]:
        """
        Map a single state from native representation to table index.
        :param states: State in native representation
        :return: Table index
        """
        mapped_state = []
        for feature in policy.domain_obs.items:
            # Get individual domain item state and map it to the requested interpretation
            domain_state = policy.domain_obs.get_item_view_by_item(states, feature)
            domain_mapped_state = FeatureConversions.as_index(feature, domain_state)
            mapped_state.append(domain_mapped_state)
        return np.asarray(mapped_state)
