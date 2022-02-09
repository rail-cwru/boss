# __init__.py
"""
Algorithm module.

Contains Reinforcement Learning Algorithms,
	which update policies.

	Compatibility should depend on domain dtype (continuous, discrete) and domain shape.
	It might be possible to consider other dependencies too.

	Some algorithms use Model.
TODO more documentation?
TODO I think we agreed on abstract classes should be exported?
"""
from .base import Algorithm
from .TemporalDifference import TemporalDifference
from .SARSA import SARSA
from .QLearning import QLearning
from .Reinforce import Reinforce
from .A2C import A2C