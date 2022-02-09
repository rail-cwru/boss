import numpy as np

from . import PolicySampler
from config import Config
from common import Properties
from domain import ObservationDomain, ActionDomain, DiscreteActionDomain

class DiscretePolicySampler(PolicySampler):

    @classmethod
    def property_helper(cls) -> Properties:
        return Properties(use_discrete_action_space=True)

    def __init__(self, config: Config, domain_obs: ObservationDomain, domain_act: ActionDomain):
        """
        Initialize a new DiscretePolicySampler (abstract)

        Discrete policy sampler samples an index from a vector of action possibilities
        raise NotImplementedError
        """
        super().__init__(config, domain_obs, domain_act)

        # Set action size from ActionDomain
        if isinstance(domain_act, DiscreteActionDomain):
            self.action_size = domain_act.full_range
        else:
            action_range = domain_act.get_action_range()
            self.action_size =[a_range.max() - a_range.min() for a_range in action_range]

    
    def sample(self, fa_values: np.ndarray):
        action, action_probs = self._sample_method(fa_values)
        return self._convert_sample(action, action_probs)
            
    def raw_sample(self, fa_values: np.ndarray):
        return self._sample_method(fa_values)

    def _convert_sample(self, action: np.ndarray, action_probs: np.ndarray) -> (int, np.ndarray):
        if self.domain_act.is_compound:
            raw_action = self.domain_act.extract_sub_actions(action)
        else:
            raw_action = [action]

        # Aggregate probabilities to calculate distribution by action
        raw_action_probs = {i:{} for i in range(len(raw_action))}
        for i in range(len(action_probs)):
            p = action_probs[i]

            if self.domain_act.is_compound:
                a = self.domain_act.extract_sub_actions(i)
            else:
                a = [i]

            for j, j_action in enumerate(a):
                if not j_action in raw_action_probs[j].keys():
                    raw_action_probs[j][j_action] = 0
                raw_action_probs[j][j_action] += p

        return raw_action, raw_action_probs

    @property
    def num_learned_parameters(self):
        return self.action_size

