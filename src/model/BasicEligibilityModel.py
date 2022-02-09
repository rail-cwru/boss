from typing import Dict, Callable, Tuple

import numpy as np

from common.properties import Properties
from config import Config
from model import ElementModel
from model.TrajectoryModel import TrajectoryModel


class BasicEligibilityModel(TrajectoryModel):
    """
    Eligibility and Old-Q storing model for implementing True Online TD-Lambda.
    """

    def __init__(self,
                 config: Config,
                 trace_shape: Tuple):
        """
        Create a new EligibilityModel (concrete)
        
        This model keeps track of eligibility traces at each time step
        """
        super().__init__(config)
        self.shape = trace_shape
        # Eligibility data
        self.__eligibility = np.zeros((config.episode_max_length + 1,) + trace_shape)
        # How many eligibilities have been appended so far
        self.n_eligs = 0
        # Old Q Data
        self.__old_q = [0]

    def add(self, value):
        self.domain_assertion(value)
        self.__eligibility[self.n_eligs] = value
        self.n_eligs += 1

    def domain_assertion(self, value):
        assert value.shape == self.shape, "Trace is incorrect shape"

    def reset(self):
        self.__eligibility.fill(0)
        self.__old_q = [0]
        self.n_eligs = 0

    @property
    def eligibilities(self) -> np.ndarray:
        return self.__eligibility[:self.n_eligs]

    @property
    def old_q_curr(self):
        return self.__old_q

    @old_q_curr.setter
    def old_q_curr(self, value):
        if isinstance(value, int):
            print('issue')
        self.__old_q = value

    @staticmethod
    def get_name():
        return "eligibility_model"

