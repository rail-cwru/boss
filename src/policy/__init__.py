# __init__.py
"""
Policy module.

Includes policies, which are stateful objects that
	represent the (usually) learned mapping from 
	agent observations to agent actions.
"""
from .base import Policy
from .FuncApproxPolicy import FuncApproxPolicy
from .TabularPolicy import TabularPolicy
from .PureStochasticPolicy import PureStochasticPolicy
from .function_approximator import *
from .policy_sampler import *
