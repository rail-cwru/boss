import numpy as np

from common.vecmath import argmax_random_tiebreak
from . import DiscretePolicySampler
from config import Config
from domain import ObservationDomain, ActionDomain

# To prevent log(0)
EPSILON = 1e-10

class DiscreteMax(DiscretePolicySampler):

    def __init__(self, config: Config, domain_obs: ObservationDomain, domain_act: ActionDomain):
        """
        Instantiate DiscreteMax sampling policy (concrete)

        Returns the index of maximum over values
        """

        super().__init__(config, domain_obs, domain_act)

    def _sample_method(self, fa_values: np.ndarray) -> (int, np.ndarray):
        """Return index of maximum valued action"""
        action = argmax_random_tiebreak(fa_values)

        num_values =len(fa_values)
        action_probs = np.full((num_values), EPSILON)
        action_probs[action] = 1.0 - ((num_values-1) * EPSILON)

        return action, action_probs

    def eval(self, fa_values: np.ndarray) -> np.ndarray:
        return fa_values

    def update_params(self):
        pass
