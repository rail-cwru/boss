from typing import Dict, List, Tuple
import numpy as np
import random

from domain.conversion import FeatureConversions
from policy.function_approximator.LinearNpy import LinearNpy
from domain import ObservationDomain, ActionDomain
from config import Config, ConfigItemDesc, ConfigDesc
from policy.function_approximator.basis_function.ExactBasis import ExactBasis
import math
import time


class SimpleLinear(LinearNpy):

    @classmethod
    def get_class_config(cls) -> List['ConfigItemDesc']:
        return [
            ConfigItemDesc('use_bias', lambda x: x is bool,
                           info='Whether to learn and use bias term for linear function.'),
            ConfigItemDesc('basis_function', lambda x: x is str, info='basis function for function approximation'
                      )
            #module_package='policy.function_approximator.basis_function'
        ]

    def __init__(self, config: Config, domain_obs: ObservationDomain, output_size: int):
        super().__init__(config, domain_obs, output_size)
        num_states = [domain_item.num_values() for domain_item in domain_obs.items]

        self.init_basis(num_states, output_size)
        self.use_bias = self.config.use_bias
        self.num_actions = output_size
        self.output_size = 1
        self.domain_obs = domain_obs

        self.weights = np.random.normal(0, 0.1, (self.output_size, self.basis_function.size()))[0]
        if self.use_bias:
            self.bias = np.random.normal(0, 0.1, self.output_size)
        else:
            self.bias = np.zeros(self.output_size)

    def init_basis(self, num_states, output_size):
        """
        Initialize basis function from config, currently only accepts ExactBasis

        :param num_states:
        :param output_size:
        :return:
        """
        if self.config.basis_function == 'ExactBasis':
            self.basis_function = ExactBasis(np.asarray(num_states), output_size)
        else:
            raise NotImplementedError("Only Exact Supported")

    def update(self, data: np.array):
        raise Exception('Simple Linear does not support this type of update')

    def eval(self, states: np.ndarray) -> np.ndarray:
        """
        Returns the value of all possible actions for a given state

        :param states: a list of states to evaluate
        :return:
        """
        mapped_state = np.asarray(self._map_table_indices(states))
        result = np.zeros(self.num_actions)
        for i in range(self.num_actions):
            result[i] = self.eval_action(mapped_state, i)
        return result

    def get_weights(self) -> np.ndarray:
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def reset_weights(self):
        self.weights = np.random.normal(0, 0.1, (self.output_size, self.basis_function.size()))[0]

    def eval_action(self, state, action):
        """
        Returns the value of an action for a given state
        """
        if action < 0 or action >= self.num_actions:
            raise IndexError('action must be in range [0, num_actions)')
        # TODO No neg vals?
        if isinstance(self.basis_function, ExactBasis):
            phi, ind = self.basis_function.evaluate2(state, action)
            return self.weights[ind]

        return self.weights.dot(self.basis_function.evaluate(state, action))

    def eval_action2(self, state, action):
        """
        Returns the value of an action for a given state
        """
        if action < 0 or action >= self.num_actions:
            raise IndexError('action must be in range [0, num_actions)')
        # TODO No neg vals?
        if isinstance(self.basis_function, ExactBasis):
            phi, ind = self.basis_function.evaluate2(state, action)
            return self.weights[ind], phi, ind

        return self.weights.dot(self.basis_function.evaluate(state, action))

    def best_action2(self, state):
        """
        Returns the highest value action for a given state
        :param state: state to eval
        :return: the number of the best primitive action
        """

        q_values = [[self.weights[ind], ind] for ind in state]
        f = lambda i: q_values[i][0]
        ind = max(range(len(q_values)), key=f)

        best_action = ind
        best_ind = int(q_values[ind][1])
        return best_action, best_ind

    def best_action(self, state):
        """
        Returns the highest value action for a given state
        :param state: state to eval
        :return: the number of the best primitive action
        """
        # This is the slowest
        q_values = [self.eval_action2(state, action)
                    for action in range(self.num_actions)]

        best_q = float('-inf')
        best_actions = []

        for action, q_value in enumerate(q_values):
            if q_value > best_q:
                best_actions = [action]
                best_q = q_value
            elif math.isclose(q_value, best_q):
                best_actions.append(action)

        best_action = random.choice(best_actions)
        return best_action

    def _map_table_indices(self, states: np.ndarray) -> Tuple[int, ...]:
        """
        Map a single state from native representation to table index.
        :param states: State in native representation
        :return: Table index
        """
        mapped_state = []
        for feature in self.domain_obs.items:
            # Get individual domain item state and map it to the requested interpretation
            domain_state = self.domain_obs.get_item_view_by_item(states, feature)
            domain_mapped_state = FeatureConversions.as_index(feature, domain_state)
            mapped_state.append(domain_mapped_state)
        return tuple(mapped_state)
