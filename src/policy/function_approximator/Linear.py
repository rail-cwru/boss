from typing import List
from torch.nn import Sequential
from torch.nn import Linear as PyTorchLinear
import torch

from config import Config, ConfigItemDesc
from policy.function_approximator.pytorch_fa import AbstractPyTorchFA


class Linear(AbstractPyTorchFA):
    """
    A Linear Function Approximator.
    """
    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return AbstractPyTorchFA.get_class_config() + [
            ConfigItemDesc('use_bias', lambda x: x is bool,
                           info='Whether to learn and use bias term for linear function.')
        ]

    def _create_model(self) -> Sequential:
        use_bias = self.config.use_bias

        layers = []
        i_layer = PyTorchLinear(self.input_size, self.output_size, bias=use_bias)
        layers.append(i_layer)

        model = Sequential(*layers)
        return model
