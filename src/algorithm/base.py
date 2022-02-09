import abc
import numpy as np
from typing import List, TYPE_CHECKING

from config.config import ConfigItemDesc
from policy import Policy
from config import Config, checks
from algorithm.ModelHandler import ModelHandler
from model import Model
from domain import ActionDomain

if TYPE_CHECKING:
    from common import Trajectory


class Algorithm(ModelHandler):

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return [
            ConfigItemDesc(name='discount_factor',
                           check=lambda gamma: 0 <= gamma <= 1,
                           info='Discount factor of the MDP as a float in [0, 1].'),
            ConfigItemDesc(name='memory_size',
                           check=checks.positive_integer,
                           info='Total memory size to store history',
                           optional=True,
                           default=1),
            ConfigItemDesc(name='batch_size',
                           check=checks.positive_integer,
                           info='Size of batch to sample from memory and train on.',
                           optional=True,
                           default=1),
            ConfigItemDesc(name='update_interval',
                           check=checks.positive_integer,
                           info='Number of steps/trajectories to observe before learning.',
                           optional=True,
                           default=1)
        ]

    def __init__(self, config: Config):
        """
        Initialize an Algorithm (abstract)

        Algorithm is an Abstract Class which is responsible for updating the Policy object.

        Algorithm is stateless and thus all methods are static
        """

        self.global_config: Config = config
        self.config = config.find_config_for_instance(self)
        self.discount_factor: float = self.config.discount_factor

    @abc.abstractmethod
    def update(self, policy: Policy, model: Model, trajectory: 'Trajectory'):
        """
        Function (abstract) which performs the actual updating of policy. Must be implemented by Algorithm
        :param policy The policy to update
        :param model The model corresponding to the updated policy
        :param trajectory The trajectory corresponding to this policy
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def compile_policy(self, policy: Policy):
         raise NotImplementedError()

    def translate_compound_action(self, domain_act: ActionDomain, actions: np.ndarray) -> np.ndarray:
        epsiode_length = actions.shape[0]
        compound_actions = np.zeros((epsiode_length,1), dtype=np.int)
        for i, action in enumerate(actions):
            compound_actions[i] = domain_act.make_compound_action(action)

        return compound_actions

