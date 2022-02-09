import abc
import numpy as np

from typing import List, Any

from common import Properties
from common.vecmath import argmax_random_tiebreak
from config import Config, AbstractModuleFrame, ConfigItemDesc
from domain import ObservationDomain, ActionDomain


class Policy(AbstractModuleFrame):

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return []

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
        # TODO: Theres a call in trajectory that requires these objects to be on policy?
        self.config = config.find_config_for_instance(self)
        self.domain_obs = domain_obs
        self.domain_act = domain_act

    @abc.abstractmethod
    def get_action_probs(self, states: np.ndarray):
        """
        Gets action probabilities given the current state
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_actions(self, states: np.ndarray, use_max: bool) -> np.ndarray:
        """
        Takes observations (called states here) and return chosen action(s)
        :param states: The observations for this step
        :param use_max: Evaluate actual learned optimal policy. That is, no exploration
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def eval(self, states: np.ndarray) -> np.ndarray:
        """
        Should run the function approximator on a feature vector
        :param states
        :return Evaluation of function approximator
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def make_feature_vectors(self, states: np.ndarray) -> np.ndarray:
        """Performs policy-specific transformations on observed features if applicable."""
        raise NotImplementedError()

    @abc.abstractmethod
    def update_with_dataset(self, states: np.ndarray, actions: np.ndarray, targets: np.ndarray):
        """Defer call to however function approximator/table decides to update itself"""
        raise NotImplementedError()

    @abc.abstractmethod
    def compile(self, loss_func, learning_rate:float):
        raise NotImplementedError()
