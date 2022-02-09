import abc
from typing import Dict, Callable, List

import numpy as np

from common.properties import Properties
from common.PropertyClass import PropertyClass
from config import Config
from config.moduleframe import AbstractModuleFrame
from config.config import ConfigItemDesc
from domain import ObservationDomain, DiscreteActionDomain, ActionDomain


class PolicySampler(PropertyClass):

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return []

    def __init__(self, config: Config, domain_obs: ObservationDomain, domain_act: ActionDomain):
        """
        Initialize a new PolicySampler (abstract)

        Policy sampler takes action domains values from
        raise NotImplementedError
        """
        self.config = config.find_config_for_instance(self)
        self.domain_obs = domain_obs
        self.domain_act = domain_act

    @property
    @abc.abstractmethod
    def num_learned_parameters(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def sample(self, fa_values: np.ndarray) -> (np.ndarray, np.ndarray):
        raise NotImplementedError()

    @abc.abstractmethod
    def eval(self, fa_values: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abc.abstractmethod
    def _sample_method(self, fa_values: np.ndarray) -> (int, np.ndarray):
        """
        Handels the actual sampling for a vector of values
        :param x: numpy array of values
        :return: index of sample (discrete) or value sampled (continuous) and probability distribution
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def update_params(self):
        raise NotImplementedError()