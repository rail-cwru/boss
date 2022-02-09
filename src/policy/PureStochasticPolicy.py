import abc
import numpy as np

from typing import List, Any

from common import Properties
from common.vecmath import argmax_random_tiebreak
from config import Config, AbstractModuleFrame, ConfigItemDesc
from domain import ObservationDomain, ActionDomain
from .base import Policy

class PureStochasticPolicy(Policy):
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
        self.action_range = self.domain_act.get_action_range()

    @classmethod
    def properties(cls) -> Properties:
        """
        Return the compatibility properties of the class.
        This must be implemented
        """
        return Properties(use_function_approximator=False)
        
    def get_action_probs(self, states: np.ndarray):
        """
        Gets action probabilities given the current state
        """
        if self.domain_act.discrete:
            num_actions = self.action_range.stop - self.action_range.start

            action_probs = np.full((num_actions), 1.0 / num_actions)
            return action_probs
        else:
            # Return truncated pdf
            action_range = self.action_range.stop - self.action_range.start
            def pdf(x):
                if x < self.action_range.start or x > self.action_range.stop:
                    return 0.0
                else:
                    return 1.0 / action_range
            return pdf

    def get_actions(self, states: np.ndarray, use_max: bool) -> np.ndarray:
        """
        Takes observations (called states here) and return chosen action(s)
        :param states: The observations for this step
        :param use_max: Evaluate actual learned optimal policy. That is, no exploration
        """

        # Ignore max value and always sample
        prob_dist = self.get_action_probs(states)

        if self.domain_act.discrete:
            return np.random.choice(len(prob_dist), 1, p=prob_dist)
        else:
            # Return sample from distribution
            return np.random.uniform(self.action_range.start, self.action_range.stop, 1)

    def eval(self, states: np.ndarray) -> np.ndarray:
        """
        Should run the function approximator on a feature vector
        :param states
        :return Evaluation of function approximator
        """
        # Since purely stochastic, only estimation is uniform distribution
        return self.get_action_probs(states)

    def make_feature_vectors(self, states: np.ndarray) -> np.ndarray:
        """Performs policy-specific transformations on observed features if applicable."""
        return states

    def update_with_dataset(self, data: np.array):
        """Defer call to however function approximator/table decides to update itself"""
        pass

    def serialize(self):
        """Return format that can be saved on disk."""
        return None

    def load(self, vals: Any):
        """Load from serialized format"""
        pass

    def compile(self, loss_func, learning_rate:float):
        pass