from typing import List, Tuple
from torch.nn import Sequential, Linear, Tanh, Module
import torch

from config import Config, ConfigItemDesc, checks
from policy.function_approximator.pytorch_fa import AbstractPyTorchFA


class MultilayerPerceptron(AbstractPyTorchFA):

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return AbstractPyTorchFA.get_class_config() + [
            ConfigItemDesc('structure', lambda l: l and checks.positive_int_list(l),
                           info='Number of inputs for each layer. First layer should be the state space, '
                           'while the last should be the input into the function approximator.')
        ]

    def _create_model(self):
        # TODO: Check if pytorch has sign activation, so tanh is used to approximate a perceptron
        hidden_layers = self.config.structure

        layers = []
        i_input = self.input_size
        for i_output in hidden_layers:
            # Add linear layer
            i_layer = Linear(i_input, i_output)
            layers.append(i_layer)

            # Add activation layer if not None
            i_activation = Tanh()
            layers.append(i_activation)

            # Replace input
            i_input = i_output

        # Add last layer
        final_output = Linear(i_input, self.output_size)
        layers.append(final_output)

        model = Sequential(*layers)
        return model
            

