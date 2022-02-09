import abc
from typing import Dict, List
import numpy as np

from config import Config, ConfigItemDesc
from domain import ObservationDomain, ActionDomain
from common.PropertyClass import PropertyClass


class FunctionApproximator(PropertyClass):

    def __init__(self,
                 config: Config,
                 domain_obs: ObservationDomain,
                 output_size: int):
        """Initialize a new function approximator (abstract)

        A function approximator is an object which takes some vector and outputs a number.

        They include things like a linear regressor, neural network, or more.
        """
        self.config = config.find_config_for_instance(self)

        self.domain_obs = domain_obs
        # self.domain_act = domain_act

        # Shape of input and output # TODO make sure this is sensible - probably have obsd and actd have .size
        self.input_size = np.prod(domain_obs.shape)
        self.output_size = output_size

    @abc.abstractmethod
    def update(self, data: np.array):
        raise NotImplementedError()

    @abc.abstractmethod
    def eval(self, states: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def get_variable_vals(self) -> Dict[str, np.ndarray]:
        """
        Gets current values of parameters
        :return: values of parameters for model by policy
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def set_variable_vals(self, vals: Dict):
        """
        Sets current value of parameters
        :param vals: Dictionary of values to set parameters with
        :return: None
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def compile(self, loss_func, learning_rate):
        raise NotImplementedError()
