import abc

from config import Config
from common import Properties
from . import PolicySampler


class PyTorchPolicySampler(PolicySampler):
    @classmethod
    def property_helper(cls) -> Properties:
        return Properties(pytorch=True)

    @abc.abstractmethod
    def sample_tensor(self, fa_output):
        """
        This returns the FA pytorch output object computation for use within the pytorch system
        """
        raise NotImplementedError
