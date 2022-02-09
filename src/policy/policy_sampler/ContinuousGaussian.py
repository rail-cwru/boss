from typing import List
import numpy as np
from torch.distributions.normal import Normal

from common import Properties
from config import Config, ConfigItemDesc
from . import ContinuousPolicySampler, PyTorchPolicySampler
from domain import ObservationDomain, ActionDomain

class ContinuousGaussian(ContinuousPolicySampler, PyTorchPolicySampler):
    """
    Gaussian Exploration for continuous action space
    """

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return ContinuousPolicySampler.get_class_config() + PyTorchPolicySampler.get_class_config() + [
            ConfigItemDesc('parameterize_std', lambda s: s == True or isinstance(s, float),
                           info='The standard deviation to use. May be "true" to allow it to be learned.')
        ]

    def __init__(self, config: Config, domain_obs: ObservationDomain, domain_act: ActionDomain):
        """Sets standard deviation to use for gaussian"""
        ContinuousPolicySampler.__init__(self, config, domain_obs, domain_act)
        PyTorchPolicySampler.__init__(self, config, domain_obs, domain_act)
        
        if self.config.parameterize_std:
            self.std_dev = None
        else:
            self.std_dev = self.config.parameterize_std

        # TODO: This needs to be truncated based on action space
        self.min = 0
        self.max = 1

    @property
    def num_learned_parameters(self):
        if self.std_dev is None:
            return 2
        else:
            return 1

    def _sample_method(self, fa_values: np.ndarray) -> (int, np.ndarray):
        # TODO: For all continuous policy samplers, whats the expected probability output
        mean = fa_values[0]
        if self.std_dev is None:
            std_dev = fa_values[1]
        else:
            std_dev = self.std_dev

        action = random_normal_variable((1,1), mean, std_dev).numpy().flatten()
        action_probs = np.ones((1,1))

        return action, action_probs

    def tensor_sample(self, fa_output):
        #TODO: Need split tensor into variables to use in random_normal, otherwise need to construct 
        # custom tensor to do the sampling
        mean = fa_output[0]
        if self.std_dev is None:
            std_dev = fa_output[1]
        else:
            std_dev = self.std_dev

        dist = Normal(mean, std_dev)
        return dist

    def eval(self, fa_values: np.ndarray) -> np.ndarray:
        mean = fa_values[0]
        if self.std_dev is None:
            std_dev = fa_values[1]
        else:
            std_dev = self.std_dev

        dist = Normal(mean, std_dev)
        return dist.sample()

    def update_params(self):
        pass
