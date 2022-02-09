# __init__.py
"""
Policy module.

Includes policies, which are stateful objects that
	represent the (usually) learned mapping from 
	agent observations to agent actions.
"""

from .base import PolicySampler
from .DiscretePolicySampler import DiscretePolicySampler
from .ContinuousPolicySampler import ContinuousPolicySampler
from .PyTorchPolicySampler import PyTorchPolicySampler
from .DiscreteEGreedy import DiscreteEGreedy
from .DiscreteMax import DiscreteMax
from .DiscreteBoltzmann import DiscreteBoltzmann
from .ContinuousGaussian import ContinuousGaussian
