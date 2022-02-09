from typing import Dict, Callable

from config import Config
from common.properties import Properties
from model.TrajectoryModel import TrajectoryModel


class ActionHistoryModel(TrajectoryModel):

    @classmethod
    @property
    def config_signature(cls) -> Dict[str, Callable]:
        sig = {
            # Any configurable fields REQUIRED OF ALL Models should go here.
        }
        return sig

    @classmethod
    @property
    def properties(cls) -> Properties:
        """
        Return the compatibility properties of the class.

        This must be implemented
        """
        return Properties()

    def __init__(self,
                 config: Config,
                 action_range: range):
        """
        Create a new EligibilityModel (concrete)
        
        This model keeps track of eligibility traces at each time step
        """
        super().__init__(config)

        self.range = action_range

    def domain_assertion(self, element):
        if not (self.range.start <= element <= self.range.stop):
            raise ValueError("Action outside possible actions: " + str(element))

    @staticmethod
    def get_name():
        return "action_history"
