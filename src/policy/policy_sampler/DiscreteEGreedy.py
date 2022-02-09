import numpy as np
from typing import List

from common.vecmath import argmax_random_tiebreak
from . import DiscretePolicySampler
from config import Config, checks, ConfigItemDesc
from common import Properties
from domain import ObservationDomain, ActionDomain

class DiscreteEGreedy(DiscretePolicySampler):

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return DiscretePolicySampler.get_class_config() + [
            ConfigItemDesc('epsilon', 
                           check=checks.unit_float,
                           info='Epsilon to use for epsilon-greedy sampling.'),
            ConfigItemDesc('decay', 
                           check= lambda x: checks.unit_float(x) and x > 0,
                           info='Decay rate for epsilon to use for epsilon-greedy sampling.',
                           optional=True,
                           default=1),
            ConfigItemDesc('min_epsilon', 
                           check= lambda x: checks.unit_float(x) and x > 0,
                           info='Minimum epsilon to not decay past, to use for epsilon-greedy sampling.',
                           optional=True,
                           default=0)
        ]

    def __init__(self, config: Config, domain_obs: ObservationDomain, domain_act: ActionDomain):
        """
        Initialize new Epsilon Greedy sampler (concrete)

        Epsilon Greedy is a method in which the best predicted option is chosen with probability epsilon
        and each other option is chosen from uniformly with total probability (1-epsilon)

        Constructor Iniitializes epsilon parameter
        """
        super().__init__(config, domain_obs, domain_act)
        self.epsilon = self.config.epsilon
        self.decay = self.config.decay
        self.min_epsilon = self.config.min_epsilon
        np.random.seed()

    def _sample_method(self, fa_values: np.ndarray) -> (int, np.ndarray):
        """Sample using epsilon greedy policy"""
        
        num_values = len(fa_values)
        greedy_action = argmax_random_tiebreak(fa_values)

        num_non_greedy_actions = float(num_values - 1)
        action_probs = np.full(len(fa_values), self.epsilon / num_non_greedy_actions)
        action_probs[greedy_action] = 1.0 - self.epsilon

        # Choose action
        action = np.random.choice(num_values, 1, p=action_probs)[0]

        return action, action_probs

    def update_params(self):
        self.epsilon *= self.decay

        if self.epsilon < self.min_epsilon:
            self.epsilon = self.min_epsilon

    def eval(self, fa_values: np.ndarray) -> np.ndarray:
        return fa_values
