from typing import TYPE_CHECKING

from algorithm import Algorithm
from model import Model
from policy import Policy
from common.properties import Properties
from config import Config

if TYPE_CHECKING:
    from common.trajectory import Trajectory


class NoAlgorithm(Algorithm):
    """
    NoAlgorithm does not perform policy updating.

    When possible, use callbacks that modify the "learn" controller flag instead.
    """

    @classmethod
    def properties_helper(cls):
        return Properties(use_function_approximator=True)

    def __init__(self, config: Config):
        super().__init__(config)

    def compile_policy(self, policy: Policy):
        pass

    def update(self, policy: Policy, model: Model, trajectory: 'Trajectory') -> Policy:
        """
        This algorithm doesn't actually learn so update does nothing
        """
        pass
