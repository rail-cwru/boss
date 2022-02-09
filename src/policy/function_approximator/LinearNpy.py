from typing import Dict, List
import numpy as np

from common import TrainingSamples, Properties

# TODO harmful to import named module like this
from domain.DiscreteActionFeatureDomain import DiscreteActionFeatureDomain
from model.BasicEligibilityModel import BasicEligibilityModel
from policy.function_approximator import FunctionApproximator
from domain import ObservationDomain, ActionDomain
from config import Config, ConfigItemDesc

""" 
TODO: This needs to be thought through more, since gradients need to be defined for policy samplers for algs like REINFORCE. 
"""

class LinearNpy(FunctionApproximator):
    @classmethod
    def properties_helper(cls):
        return Properties(pytorch=True)

    @classmethod
    def get_class_config(cls) -> List['ConfigItemDesc']:
        return [
            ConfigItemDesc('use_bias', lambda x: x is bool,
                           info='Whether to learn and use bias term for linear function.')
        ]

    def __init__(self, config: Config, domain_obs: ObservationDomain, output_size: int):
        super().__init__(config, domain_obs, output_size)
        self.use_bias = self.config.use_bias

        self.weights = np.random.normal(0, 0.1, (self.output_size, self.input_size))
        if self.use_bias:
            self.bias = np.random.normal(0, 0.1, self.output_size)
        else:
            self.bias = np.zeros(self.output_size)

    def update(self, states: np.ndarray, targets: np.ndarray):
        raise Exception('Numpy version of Linear FA is not complete.')

        n = states.shape[0]

        y_pred = np.zeros((n, self.output_size))
        for i in range(n):
            y_pred[i] = self.eval(states[i])
        
        loss = self.loss(y_pred, targets)

        # Update weights
        sampler_gradient = 0 #TODO
        m_gradient = sampler_gradient * targets 
        self.weights += self.learning_rate * m_gradient
        
        # Update bias
        if self.use_bias:
            b_gradient = sampler_gradient * targets
            self.bias += self.learning_rate * b_gradient

    def eval(self, states: np.ndarray) -> np.ndarray:
        result = np.zeros((self.output_size))
        for i in range(self.output_size):
            result[i] = np.dot(self.weights[i,:], states) + self.bias[i]

        return result

    def compile(self, loss_func, learning_rate):
        self.loss = loss_func
        self.learning_rate = learning_rate

    def get_variable_vals(self) -> Dict[str, np.ndarray]:
        """
        Gets current values of parameters
        :return: values of parameters for model by policy
        """
        val_dict = {
            'weights': self.weights,
            'bias': self.bias
        }
        return val_dict

    def set_variable_vals(self, vals: Dict):
        """
        Sets current value of parameters
        :param vals: Dictionary of values to set parameters with
        :return: None
        """
        self.weights = vals['weights']
        self.bias = vals['bias']

