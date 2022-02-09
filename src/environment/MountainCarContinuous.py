from typing import Dict, Union, List, Tuple, Any

import gym as gym
import numpy as np


from domain.actions import RealAction
from domain.observation import ObservationDomain
from domain.ActionDomain import ActionDomain
from domain.features import RealFeature
from common.properties import Properties
from config import Config
from config.config import ConfigItemDesc
from environment import Environment


class MountainCarContinuous(Environment):
    """
    wrapper for MountainCarContinuous
    https://github.com/openai/gym/wiki/MountainCarContinuous-v0
    """

    @classmethod
    def get_class_config(cls) -> List['ConfigItemDesc']:
        return []

    @classmethod
    def properties(cls) -> Properties:
        return Properties(use_discrete_action_space=False,
                          use_joint_observations=False,
                          use_agent_deletion=False,
                          use_agent_addition=False)
    
    def __init__(self, config: Config):
        self.wrapped_env = gym.make('MountainCarContinuous-v0')
        #only one agent, and one class.
        self._agent_id_list = [0]
        self._agent_class_list = [0]
        self._agent_class_map = {0:0}
        super(MountainCarContinuous, self).__init__(config)

    def set_initial_seed(self, seed: int):
        self.wrapped_env.seed(seed)

    def get_seed_state(self):
        return self.wrapped_env.np_random.get_state()

    def set_seed_state(self, seed_state):
        return self.wrapped_env.np_random.set_state(seed_state)

    def update(self, actions: Dict[int, np.ndarray]) -> Dict[int, int]:
        action = actions[0]
        self._current_observation, reward, done, info = self.wrapped_env.step(action[0])
        if done:
            self.done = True
        return {0: reward if not self.done else -1}

    def observe(self, obs_groups: List[Tuple[int, ...]] = None) -> Dict[Union[int, Tuple[int, ...]], np.ndarray]:
        return {0: self._current_observation}

    def _reset_state(self, visualize: bool = False) -> Any:
        self._current_observation = self.wrapped_env.reset()
        return self.wrapped_env

    def _create_observation_domains(self, config) -> Dict[Union[str, int], ObservationDomain]:
        upper_domain_bound = self.wrapped_env.observation_space.high
        lower_domain_bound = self.wrapped_env.observation_space.low

        cart_position = RealFeature('car_position', lower_domain_bound[0], upper_domain_bound[0])
        cart_velocity = RealFeature('car_velocity', lower_domain_bound[1], upper_domain_bound[1])

        self._observation_domain = ObservationDomain([cart_position, cart_velocity], num_agents=1)
        return {0: self._observation_domain}

    @property
    def agent_class_map(self) -> Dict[int, int]:
        return self._agent_class_map

    @property
    def agent_class_list(self) -> List[str]:
        return self._agent_class_list

    @property
    def agent_id_list(self) -> List[int]:
        return self._agent_id_list

    def _create_action_domains(self, config) -> Dict[Union[str, int], ActionDomain]:
        upper_domain_bound = self.wrapped_env.observation_space.high
        lower_domain_bound = self.wrapped_env.observation_space.low

        push_car = RealAction('push_car', lower_domain_bound[0], upper_domain_bound[0])
        self._action_domain = ActionDomain([push_car], 1)
        return {0: self._action_domain}

    def visualize(self):
        self.wrapped_env.render()
