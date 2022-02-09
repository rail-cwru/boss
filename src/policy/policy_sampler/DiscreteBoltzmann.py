from typing import List
import numpy as np
from torch.nn import Softmax
import torch
# from scipy.special import softmax

from common import Properties
from config import Config, ConfigItemDesc, checks
from . import DiscretePolicySampler, PyTorchPolicySampler
from domain import ObservationDomain, ActionDomain

# To prevent log(0)
EPSILON = 1e-10

class DiscreteBoltzmann(DiscretePolicySampler, PyTorchPolicySampler):

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return [
            ConfigItemDesc('temperature', checks.positive_float, info='Sampling temperature. Typically 1.0.')
        ]

    def __init__(self, config: Config, domain_obs: ObservationDomain, domain_act: ActionDomain):
        DiscretePolicySampler.__init__(self, config, domain_obs, domain_act)
        PyTorchPolicySampler.__init__(self, config, domain_obs, domain_act)
        self.temperature = self.config.temperature
        self.softmax_tensor = Softmax(dim=1)

    # @profile
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Takes of the softmax of function approximator values
        :param x:
        :return: numpy array representing probability distribution
        """
        if self.temperature != 1.0:
            x = x / self.temperature

        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def eval(self, fa_values: np.ndarray) -> np.ndarray:
        return self._softmax(fa_values)

    def _sample_method(self, fa_values: np.ndarray) -> (int, np.ndarray):
        action_probs = self._softmax(fa_values)
        num_actions = action_probs.shape[0]

        # Choose action
        action = np.random.choice(num_actions, p=action_probs)

        # Adjust in case of 0
        adjusted = False
        for i in range(num_actions):
            if action_probs[i] == 0:
                action_probs[i] = EPSILON
                adjusted = True

        if adjusted:
            prob_sum = np.sum(action_probs)
            action_probs /= prob_sum

        return action, action_probs

    def sample_tensor(self, fa_output: torch.Tensor):
        # Apply temperature
        parameters = fa_output
        if self.temperature != 1.0:
            parameters = torch.div(parameters, self.temperature) 

        prob = self.softmax_tensor(parameters)

        # Apply epsilon to 0 to prevent log(0)
        mask_prob = prob.clone()
        mask_prob[prob==0] = EPSILON
        mask_prob = mask_prob / mask_prob.sum(1, keepdim=True)[0]

        return mask_prob

    def update_params(self):
        pass