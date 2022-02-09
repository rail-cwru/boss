# __init__.py
"""
Environment module.

The environment specifies the entire state, and the agents in it.
It will also act as the transition function between states, as well as provide the rewards and observations
for each agent at each time step.

Environment will interact with the AgentSystem for the experiment. Environments can have multiple agents.
Whether these are coordinated, or independent agents should not affect the compatibility.
"""
from .base import Environment, HandcraftEnvironment
from .GridWorld import GridWorld
from .Cartpole import Cartpole
from .TestEnvironment import TestEnvironment
from .Drift import Drift
from .MountainCarContinuous import MountainCarContinuous
from .PredatorPrey import PredatorPrey
