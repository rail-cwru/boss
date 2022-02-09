import abc
import numpy as np

from typing import List, Any, Tuple

from common import Properties
from domain.conversion import FeatureConversions
from domain.features import FeatureType
from common.vecmath import argmax_random_tiebreak
from config import Config, AbstractModuleFrame, ConfigItemDesc, ConfigDesc
from domain import ObservationDomain, ActionDomain
from .base import Policy
from policy.policy_sampler import DiscretePolicySampler


class TabularPolicy(Policy):

    NATIVE_FEATURES = {FeatureType.BINARY, FeatureType.DISCRETE}

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return super().get_class_config() + [
            ConfigDesc(name='policy_sampler',
                       module_package='policy.policy_sampler',
                       info='Policy Sampler to use with Policy.',
                       default_configs=[])
        ]

    @classmethod
    def properties(cls) -> Properties:
        """
        Return the compatibility properties of the class.
        This must be implemented
        """
        return Properties(use_function_approximator=False)

    def __init__(self,
                 domain_obs: ObservationDomain,
                 domain_act: ActionDomain,
                 config: Config):
        """
        Instantiate a new Policy Class (abstract)
        Base class which all Policies derive
        :param domain_obs The observation domain
        :param domain_act The action domain
        :param config The config for the experiment
        """
        super().__init__(domain_obs, domain_act, config)

        if not domain_act.discrete or not domain_obs.discrete:
            raise TypeError('Tabular policy cannot be used for continuous action or observation domains.')

        # Initialize discrete policy sampler
        self.sampler: DiscretePolicySampler = self.config.policy_sampler.module_class(config, domain_obs, domain_act)

        # Initialize table
        num_actions = self.domain_act.full_range
        num_states = [domain_item.num_values() for domain_item in domain_obs.items]

        shape_tuple = (*num_states, num_actions)
        self.table = np.full(shape_tuple, 0.123)


    def get_action_probs(self, states: np.ndarray) -> np.ndarray:
        """
        Gets action probabilities given the current state
        """
        action_values = self.eval(states)
        _, prob_dist = self.sampler.sample(action_values)
        return prob_dist

    def get_actions(self, states: np.ndarray, use_max: bool) -> np.ndarray:
        """
        Takes observations (called states here) and return chosen action(s)
        :param states: The observations for this step
        :param use_max: Evaluate actual learned optimal policy. That is, no exploration
        """
        action_values = self.eval(states)
        if use_max:
            sampler_values = self.sampler.eval(action_values)
            actions = argmax_random_tiebreak(sampler_values)

            if self.domain_act.is_compound:
                raw_action = np.array([self.domain_act.extract_sub_actions(actions)])
            else:
                raw_action = np.array([actions])
        else:
            raw_action, _ = self.sampler.sample(action_values)

        # sampler_values = self.sampler.eval(action_values)
        # actions = argmax_random_tiebreak(sampler_values)
        # ra = np.array([actions])

        return raw_action

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

    def eval(self, states: np.ndarray) -> np.ndarray:
        """
        Evaluates the function approximator on a single state / feature vector.
        :param states: State/feature vectors.
        :return Evaluation of function approximator
        """
        if states.ndim == 1:
            # TODO Single state - illegitimate use
            return self.table[self._map_table_indices(states)]
        elif states.ndim > 1:
            return np.stack([self.table[self._map_table_indices(state)] for state in states], axis=0)
        else:
            raise ValueError("States was of wrong rank. Expected 1, 2, or greater, not [{}]".format(states.ndim))

    def make_feature_vectors(self, states: np.ndarray) -> np.ndarray:
        """Performs policy-specific transformations on observed features if applicable."""
        return states

    def compile(self, loss_func, learning_rate:float):
        # Store these here similar to how the FAs store them
        self.loss = loss_func
        self.learning_rate = learning_rate

    def update_with_dataset(self, data: np.array):
        """Defer call to however function approximator/table decides to update itself"""
        states, actions, targets = self.loss(self, data)
        # Apply learning rate
        targets = np.multiply(targets, self.learning_rate)
        data_set_size = states.shape[0]
        for i in range(data_set_size):
            mapped_indices = self._map_table_indices(states[i])
            table_indices = tuple(np.append(mapped_indices, actions[i]))
            self.table[table_indices] += targets[i]
            #print(self.table[table_indices])

    def h_update(self, states: np.ndarray, actions: np.ndarray, targets: np.ndarray):
        """Defer call to however function approximator/table decides to update itself"""
        data_set_size = states.shape[0]
        for i in range(data_set_size):
            mapped_indices = self._map_table_indices(states[i])
            table_indices = tuple(np.append(mapped_indices, actions[i]))
            self.table[table_indices] += targets[i]


    def update_value_fn(self, states: np.ndarray, targets: np.ndarray ):
        '''
        Used to update a value function which is a single vector
        Other method (update_with_dataset) updates multiple values
        :param states:
        :param actions:
        :param targets:
        :return:
        '''

        data_set_size = states.shape[0]
        for i in range(data_set_size):
            mapped_indices = self._map_table_indices(states[i])
            table_indices = mapped_indices
            self.table[table_indices] += targets[i]

    def serialize(self):
        """Return format that can be saved on disk."""
        return self.table

    def load(self, vals: Any):
        """Load from serialized format"""
        self.table = vals


