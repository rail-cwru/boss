import gym
from typing import Dict, Union, List, Tuple, Any
import numpy as np
from copy import deepcopy

from domain.actions import DiscreteAction, ActionType
from domain import ObservationDomain, DiscreteActionDomain, ActionDomain
from domain.features import RealFeature, DiscreteFeature, BinaryFeature
from common.properties import Properties
from config import Config, checks
from config.config import ConfigItemDesc
from environment.ConvertedDiscrete import ConvertedDiscrete


class BipedalWalker(ConvertedDiscrete):
    """
    Wrapper for Bipedal Walker using the Box2d engine
    """
    @classmethod
    def properties(cls) -> Properties:
        return Properties(use_discrete_action_space=True,
                          use_joint_observations=False,
                          use_agent_deletion=False,
                          use_agent_addition=False)

    def __init__(self, config: Config):
        self.wrapped_env = gym.make('BipedalWalker-v2')
        #only one agent, and one class.
        self._agent_id_list = [0]
        self._agent_class_list = [0]
        self._agent_class_map = {0:0}
        self._obs_after_update = None
        self.num_discrete_bins = config.environment.num_discrete_bins
        super(BipedalWalker, self).__init__(config)

    def set_initial_seed(self, seed: int):
        self.wrapped_env.env.seed(seed)

    def get_seed_state(self):
        return self.wrapped_env.env.np_random.get_state()

    def set_seed_state(self, seed_state):
        return self.wrapped_env.env.np_random.set_state(seed_state)

    @property
    def upper_action_bound(self):
        return self.wrapped_env.action_space.high

    @property
    def lower_action_bound(self):
        return self.wrapped_env.action_space.low

    def update(self, actions: Dict[int, np.ndarray]) -> Dict[int, int]:
        # Convert to continuous values
        action = self._convert_action(actions[0])

        self._obs_after_update, reward, done, info = self.wrapped_env.step(action)
        if done:
            self.done = True
        return {0: reward}

    def observe(self, obs_groups: List[Tuple[int, ...]] = None) -> Dict[Union[int, Tuple[int, ...]], np.ndarray]:
        return {0: self._obs_after_update}

    def _reset_state(self, visualize: bool = False) -> Any:
        self._obs_after_update = self.wrapped_env.reset()
        return self.wrapped_env

    def _create_observation_domains(self, config) -> Dict[Union[str, int], ObservationDomain]:
        upper_domain_bound = deepcopy(self.wrapped_env.observation_space.high)
        lower_domain_bound = deepcopy(self.wrapped_env.observation_space.low)
        num_domain_items = upper_domain_bound.shape[0]

        min_value = np.finfo(upper_domain_bound.dtype).min
        max_value = np.finfo(upper_domain_bound.dtype).max

        # Convert infinities
        for i in range(num_domain_items):
            if lower_domain_bound[i] == -float('inf'):
                lower_domain_bound[i] = min_value
            if upper_domain_bound[i] == float('inf'):
                upper_domain_bound[i] = max_value
        
        features = [
            RealFeature('hull_angle'.format(i), 0, 2 * np.pi),
            RealFeature('hull_angular_velocity'.format(i), lower_domain_bound[1], upper_domain_bound[1]),
            RealFeature('vel_x'.format(i), -1, 1),
            RealFeature('velx_y'.format(i), -1, 1),
            RealFeature('hip_joint_1_angle'.format(i), lower_domain_bound[4], upper_domain_bound[4]),
            RealFeature('hip_joint_1_speed'.format(i), lower_domain_bound[5], upper_domain_bound[5]),
            RealFeature('knee_joint_1_angle'.format(i), lower_domain_bound[6], upper_domain_bound[6]),
            RealFeature('knee_joint_1_speed'.format(i), lower_domain_bound[7], upper_domain_bound[7]),
            BinaryFeature('leg_1_ground_contact'),
            RealFeature('hip_joint_2_angle'.format(i), lower_domain_bound[9], upper_domain_bound[9]),
            RealFeature('hip_joint_2_speed'.format(i), lower_domain_bound[10], upper_domain_bound[10]),
            RealFeature('knee_joint_2_angle'.format(i), lower_domain_bound[11], upper_domain_bound[11]),
            RealFeature('knee_joint_2_speed'.format(i), lower_domain_bound[12], upper_domain_bound[12]),
            BinaryFeature('leg_2_ground_contact')
        ]

        # Add lidar readings
        for i in range(10):
            i_feature = RealFeature('lidar_{}'.format(i), lower_domain_bound[i+14], upper_domain_bound[i+14])
            features.append(i_feature)

        self._observation_domain = ObservationDomain(features, num_agents=1)
        return {0: self._observation_domain}

    def _create_action_domains(self, config) -> Dict[Union[str, int], ActionDomain]:
        hip_1 = DiscreteAction('hip_1', self.num_discrete_bins)
        knee_1 = DiscreteAction('knee_1', self.num_discrete_bins)
        hip_2 = DiscreteAction('hip_2', self.num_discrete_bins)
        knee_2 = DiscreteAction('knee_2', self.num_discrete_bins)
        self._action_domain = DiscreteActionDomain([hip_1, knee_1, hip_2, knee_2], 1)

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
