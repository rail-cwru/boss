from abc import abstractproperty
from typing import Dict, Union, List, Tuple, Any
import gym as gym
import numpy as np

from domain.actions import DiscreteAction, ActionType
from domain import ObservationDomain, DiscreteActionDomain, ActionDomain
from domain.features import RealFeature, DiscreteFeature
from common.properties import Properties
from config import Config, checks
from config.config import ConfigItemDesc
from environment import Environment


class ConvertedDiscrete(Environment):
    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return [
            ConfigItemDesc(name='num_discrete_bins',
                           check=checks.positive_integer,
                           info='Number of discrete bins to create for each continuous action space.',
                           optional=True, default=10)
        ]

    @abstractproperty
    def upper_action_bound(self):
        raise NotImplementedError()

    @abstractproperty
    def lower_action_bound(self):
        raise NotImplementedError()

    def _convert_action(self, action):
        # Convert to continuous
        steps = self.upper_action_bound - self.lower_action_bound
        steps /= self.num_discrete_bins

        continuous_action = self.lower_action_bound + action * steps
        return continuous_action

    