from . import PolicySampler
from config import Config
from common import Properties

class ContinuousPolicySampler(PolicySampler):

    @classmethod
    def property_helper(cls) -> Properties:
        return Properties(use_discrete_action_space=False)

