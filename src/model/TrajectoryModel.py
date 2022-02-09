import abc
from typing import Callable, Dict

from config import Config
from common.properties import Properties
from model import ElementModel


class TrajectoryModel(ElementModel):

    @classmethod
    def config_signature(cls) -> Dict[str, Callable]:
        sig = {
            # Any configurable fields REQUIRED OF ALL Models should go here.
        }
        return sig

    @classmethod
    def properties(cls) -> Properties:
        """
        Return the compatibility properties of the class.

        This must be implemented
        """
        return Properties()

    def __init__(self, config: Config):
        """
        Instantiate a new Trajectory Model (abstract)

        This model form is used for any model which keeps track of a value or set of values at each time step.
        In other words, it acts as an "addendum" to the official trajectory.
        """
        super().__init__(config)
        self.values = []

    def add(self, value):
        """
        Add value to list stored by trajectory
        :param value: Value to add
        :return: None
        """
        self.domain_assertion(value)
        self.values.append(value)

    @abc.abstractmethod
    def domain_assertion(self, value):
        """
        Asserts that element being added is valid
        :param value:  value to be addded
        :return: None
        """
        return NotImplementedError()

    def reset(self):
        """
        Empties model. Should be used for end of episode
        :return: None
        """
        self.values = []
