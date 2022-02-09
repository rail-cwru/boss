from typing import List, Tuple
from torch.nn import Sequential, Linear, ReLU, Tanh
import torch

from config import Config, ConfigItemDesc, checks
from policy.function_approximator.pytorch_fa import AbstractPyTorchFA

PYTORCH_ACTIVATIONS = {
    'relu': ReLU,
    'tanh': Tanh
}

class DenseNeuralNet(AbstractPyTorchFA):
    # TODO name's too general, let's change it to be more specific

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        # TODO: Add check for activations and size to be positive
        return AbstractPyTorchFA.get_class_config() + [
            ConfigItemDesc('structure', lambda x: checks.positive_int_str_tuple_list(x),
                           info='Number of inputs for each layer as well as the activation to use, first ' 
                                'layer should be the state space, while the last should be the input into the '
                                'function approximator.')
        ]

    def _create_model(self):
        hidden_layers = self.config.structure
        layers = []

        i_input = self.input_size
        for layer in hidden_layers:
            i_output, activation = layer
            
            # Add linear layer
            i_layer = Linear(i_input, i_output)
            #i_layer.weight.data.fill_(0.01)
            #i_layer.bias.data.fill_(0.01)
            layers.append(i_layer)

            # Add activation layer if not None
            if activation is not None:
                i_activation = PYTORCH_ACTIVATIONS[activation]()
                layers.append(i_activation)

            # Replace input
            i_input = i_output

        # Add last layer
        final_output = Linear(i_input, self.output_size)
        #final_output.weight.data.fill_(0.01)
        #final_output.bias.data.fill_(0.01)
        layers.append(final_output)

        model = Sequential(*layers)
        return model

