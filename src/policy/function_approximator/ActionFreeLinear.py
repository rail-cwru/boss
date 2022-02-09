import torch.nn as tnn
import torch

from domain import ObservationDomain
from domain.DiscreteActionFeatureDomain import DiscreteActionFeatureDomain
from policy.function_approximator.pytorch_fa import AbstractPyTorchFA
from config import Config


class ActionFreeLinear(AbstractPyTorchFA):
    """
    A Linear Function Approximator which assumes that
    """

    def __init__(self,
                 config: Config,
                 domain_obs: ObservationDomain,
                 output_size: int):
        assert isinstance(domain_obs, DiscreteActionFeatureDomain)
        self.action_size, self.feature_size = domain_obs.packed_shape
        assert self.action_size == output_size
        super(ActionFreeLinear, self).__init__(config, domain_obs, output_size)

    def _create_model(self, config: Config) -> tnn.Sequential:
        action_size = self.action_size
        feature_size = self.feature_size

        class Net(tnn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.w = tnn.Linear(feature_size, 1, bias=False)

            def forward(self, x: torch.Tensor):
                # [N, A*F] -> [N, A, F] -> [N, A, 1] -> [N, A]
                return self.w(x.view(-1, action_size, feature_size)).view(-1, action_size)

        return Net()
