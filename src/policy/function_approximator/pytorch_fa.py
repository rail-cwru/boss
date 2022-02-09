"""
Function approximator base class for pytorch-using FAs.
"""

import abc
from typing import List, Dict
import numpy as np
from torch.optim import ASGD, Adam, Adagrad, SGD
from torch.nn import Sequential
import torch

from common.properties import Properties
from config import Config, ConfigItemDesc
from domain import ObservationDomain, ActionDomain
from . import FunctionApproximator

PYTORCH_OPTIMIZERS = {
    'adam': Adam,
    'adagrad' : Adagrad,
    'asgd': ASGD,
    'sgd': SGD
}

DEVICES = {'cuda': torch.device('cuda'),
           'cpu': torch.device('cpu')}


class AbstractPyTorchFA(FunctionApproximator):

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return [
                ConfigItemDesc('device', lambda s: s in DEVICES,
                               info="Default cpu. Choices: " + ' or '.join([k for k in DEVICES]),
                               default='cpu',
                               optional=True),
                ConfigItemDesc('optimizer', lambda s: s in PYTORCH_OPTIMIZERS,
                               "Choice of optimizer to optimize gradients / losses with.\n"
                               "Unstable; May be subject to API change.\n"
                               "Current options are: " + ', '.join([k for k in PYTORCH_OPTIMIZERS])),
            ]

    @classmethod
    def properties_helper(cls):
        return Properties(pytorch=True)

    def __init__(self,
                 config: Config,
                 domain_obs: ObservationDomain,
                 output_size: int):
        """
        Does not define loss: That is up to the learning algorithm.
        """
        super().__init__(config, domain_obs, output_size)
        self.optimizer_cls = PYTORCH_OPTIMIZERS[self.config.optimizer]
        self.device = DEVICES[self.config.device]
        self.model = self._create_model().float()
        self.model = self.model.to(self.device)
        self.optimizer = None

    @abc.abstractmethod
    def _create_model(self) -> Sequential:
        raise NotImplementedError()

    def eval(self, states: np.ndarray) -> np.ndarray:
        x_var = torch.tensor(states, dtype=torch.float, device=self.device)

        y_var = self.eval_tensor(x_var)
        y = self.to_numpy(y_var.detach())

        y = np.reshape(y, (-1,))
        return  y

    def eval_tensor(self, states: torch.Tensor) -> torch.Tensor:
        y_var = self.model(states)
        return y_var

    def update(self, data: np.array):
        # Calculate loss
        loss = self.loss(data)

        # Zero gradient before back prop
        self.model.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()

        # Update parameters
        self.optimizer.step()

    def compile(self, loss_func, learning_rate):
        self.loss = loss_func
        self.optimizer = self.optimizer_cls(self.model.parameters(), lr=learning_rate)

    def get_variable_vals(self) -> Dict[str, np.ndarray]:
        """
        Gets current values of parameters
        :return: values of parameters for model by policy
        """
        # WARNING!!! CHANGES IN SERIALIZATION FORMAT OR FA STRUCTURE WILL BREAK COMPATIBILITY WITH OLD CODE.
        model_state_dict = self.model.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()

        # Convert to numpy
        model_numpy = {key: self.to_numpy(value) for key, value in model_state_dict.items()}

        return {'model': model_numpy, 'optimizer': optimizer_state_dict}

    def set_variable_vals(self, vals: Dict):
        """
        Sets current value of parameters
        :param vals: Dictionary of values to set parameters with
        :return: None
        """
        # WARNING!!! CHANGES IN SERIALIZATION FORMAT OR FA STRUCTURE WILL BREAK COMPATIBILITY WITH OLD CODE.
        model_state_dict = {key: torch.tensor(value, dtype=torch.float, device=self.device) for key, value in vals['model'].items()}

        self.model.load_state_dict(model_state_dict, strict=False)
        self.optimizer.load_state_dict(vals['optimizer'])

    def to_numpy(self, tensor) -> np.ndarray:
        if self.device == DEVICES['cuda']:
            return tensor.cpu().numpy()
        else:
            return tensor.numpy()
