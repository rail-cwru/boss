from typing import List, Dict
import numpy as np
from torch.nn import Sequential, Linear, ReLU, Tanh, Module, ModuleList
import torch

from common.properties import Properties
from config import Config, ConfigItemDesc, ConfigDesc, checks
from domain import ObservationDomain, ActionDomain
from policy.function_approximator.pytorch_fa import AbstractPyTorchFA
from . import FunctionApproximator

PYTORCH_ACTIVATIONS = {
    'relu': ReLU,
    'tanh': Tanh
}

class ACFunctionApproximator(AbstractPyTorchFA):
    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return AbstractPyTorchFA.get_class_config() + [
                ConfigItemDesc('actor_structure', lambda x: checks.positive_int_str_tuple_list(x),
                           info='Number of inputs for each layer as well as the activation to use, first ' 
                                'layer should be the state space, while the last should be the input into the '
                                'function approximator.'),
                ConfigItemDesc('critic_structure', lambda x: checks.positive_int_str_tuple_list(x),
                           info='Number of inputs for each layer as well as the activation to use, first ' 
                                'layer should be the state space, while the last should be the input into the '
                                'function approximator.')
            ]

    def __init__(self,
                 config: Config,
                 domain_obs: ObservationDomain,
                 output_size: int):
        """
        Class that holds two FAs for actor critic methods
        """
        super().__init__(config, domain_obs, output_size)

        # Default model is actor, create a critic
        self.critic_model = self._create_model(True).float()
        self.critic_model = self.critic_model.to(self.device)

    def _create_model(self, create_critic: bool = False):
        input_size = self.input_size
        if create_critic:
            hidden_layers = self.config.critic_structure
            output_size = 1
        else:
            hidden_layers = self.config.actor_structure
            output_size = self.output_size

        layers = []
        i_input = input_size
        for layer in hidden_layers:
            i_output, activation = layer

            # Add linear layer
            i_layer = Linear(i_input, i_output)
            layers.append(i_layer)

            # Add activation layer if not None
            if activation is not None:
                i_activation = PYTORCH_ACTIVATIONS[activation]()
                layers.append(i_activation)

            # Replace input
            i_input = i_output

        # Add last layer
        final_output = Linear(i_input, output_size)
        layers.append(final_output)

        model = Sequential(*layers)
        return model

    def eval_critic_tensor(self, states: torch.Tensor) -> torch.Tensor:
        y_var = self.critic_model(states)
        return y_var

    def get_variable_vals(self) -> Dict[str, np.ndarray]:
        """
        Gets current values of parameters
        :return: values of parameters for model by policy
        """
        val_dict = AbstractPyTorchFA.get_variable_vals(self)

        # Add critic info
        model_state_dict = self.critic_model.state_dict()
        model_numpy = {key: self.to_numpy(value) for key, value in model_state_dict.items()}
        val_dict['critic_model'] = model_numpy

        return val_dict

    def set_variable_vals(self, vals: Dict):
        """
        Sets current value of parameters
        :param vals: Dictionary of values to set parameters with
        :return: None
        """
        AbstractPyTorchFA.set_variable_vals(self, vals)

        # Set critic info
        model_state_dict = {key: torch.tensor(value, dtype=torch.float, device=self.device) for key, value in vals['critic_model'].items()}
        self.critic_model.load_state_dict(model_state_dict, strict=False)

    def update(self, data: np.array):
        # Calculate loss
        loss = self.loss(data)

        # Zero gradient before back prop
        self.model.zero_grad()
        self.critic_model.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()

        # Update parameters
        self.optimizer.step()

    def compile(self, loss_func, learning_rate):
        all_params = list(self.model.parameters()) + list(self.critic_model.parameters())
        self.loss = loss_func
        self.optimizer = self.optimizer_cls(all_params, lr=learning_rate)