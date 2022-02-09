from typing import Dict, Union, List, Tuple, Any

import gym as gym
import numpy as np

from domain import DiscreteActionDomain
from domain.actions import DiscreteAction
from domain.observation import ObservationDomain
from domain.ActionDomain import ActionDomain
from domain.features import RealFeature, DiscreteFeature
from common.properties import Properties
from config import Config
from config.config import ConfigItemDesc
from environment import Environment


class Cartpole(Environment):
    """
    wrapper for CartPole
    https://github.com/openai/gym/wiki/CartPole-v0
    """

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return []

    @classmethod
    def properties(cls) -> Properties:
        return Properties(use_discrete_action_space=True,
                          use_joint_observations=False,
                          use_agent_deletion=False,
                          use_agent_addition=False)

    def __init__(self, config: Config):
        self.wrapped_env = gym.make('CartPole-v1')
        #only one agent, and one class.
        self._agent_id_list = [0]
        self._agent_class_list = [0]
        self._agent_class_map = {0:0}
        super(Cartpole, self).__init__(config)

    def set_initial_seed(self, seed: int):
        self.wrapped_env.seed(seed)

    def get_seed_state(self):
        return self.wrapped_env.np_random.get_state()

    def set_seed_state(self, seed_state):
        return self.wrapped_env.np_random.set_state(seed_state)

    def update(self, actions: Dict[int, np.ndarray]) -> Dict[int, int]:
        action = actions[0]
        self.state, reward, done, info = self.wrapped_env.step(action[0])
        if done:
            self.done = True
        return {0: reward}

    def observe(self, obs_groups: List[Tuple[int, ...]] = None) -> Dict[Union[int, Tuple[int, ...]], np.ndarray]:
        return {0: self.state}

    def _reset_state(self, visualize: bool = False) -> Any:
        self.state = self.wrapped_env.reset()
        return self.state

    def _create_observation_domains(self, config) -> Dict[Union[str, int], ObservationDomain]:
        upper_domain_bound = self.wrapped_env.observation_space.high
        lower_domain_bound = self.wrapped_env.observation_space.low
        
        cart_position = RealFeature('cart_position', lower_domain_bound[0], upper_domain_bound[0])
        cart_velocity = RealFeature('cart_velocity', lower_domain_bound[1], upper_domain_bound[1])
        pole_angle = RealFeature('pole_angle', lower_domain_bound[2], upper_domain_bound[2])
        pole_velocity = RealFeature('pole_velocity', lower_domain_bound[3], upper_domain_bound[3])

        self._observation_domain = ObservationDomain([cart_position, cart_velocity, pole_angle, pole_velocity],
                                                     num_agents=1)
        return {0: self._observation_domain}

    def _create_action_domains(self, config) -> Dict[Union[str, int], ActionDomain]:
        push_car = DiscreteAction('push_car', self.wrapped_env.action_space.n)
        self._action_domain = DiscreteActionDomain([push_car], 1)
        return {0: self._action_domain}

    @property
    def agent_class_map(self) -> Dict[int, int]:
        return self._agent_class_map

    @property
    def agent_class_list(self) -> List[str]:
        return self._agent_class_list

    @property
    def agent_id_list(self) -> List[int]:
        return self._agent_id_list

    def visualize(self):
        self.wrapped_env.render()
