import gym
import pybulletgym
from typing import Dict, Union, List, Tuple, Any
import numpy as np
from copy import deepcopy

from domain.actions import DiscreteAction, ActionType
from domain import ObservationDomain, DiscreteActionDomain, ActionDomain
from domain.features import RealFeature, DiscreteFeature
from common.properties import Properties
from config import Config, checks
from config.config import ConfigItemDesc
from environment.ConvertedDiscrete import ConvertedDiscrete


class Hopper(ConvertedDiscrete):
    """
    Wrapper for Hopper originally from Mujoco, but using the pybullet physics engine
    """
    @classmethod
    def properties(cls) -> Properties:
        return Properties(use_discrete_action_space=True,
                          use_joint_observations=False,
                          use_agent_deletion=False,
                          use_agent_addition=False)

    def __init__(self, config: Config):
        self.wrapped_env = gym.make('HopperPyBulletEnv-v0')
        #only one agent, and one class.
        self._agent_id_list = [0]
        self._agent_class_list = [0]
        self._agent_class_map = {0:0}
        self._obs_after_update = None
        self.num_discrete_bins = config.environment.num_discrete_bins
        super(Hopper, self).__init__(config)

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
        if visualize:
            self.wrapped_env.render()
        self._obs_after_update = self.wrapped_env.reset()
        return self.wrapped_env

    def _create_observation_domains(self, config) -> Dict[Union[str, int], ObservationDomain]:
        upper_domain_bound = deepcopy(self.wrapped_env.observation_space.high)
        lower_domain_bound = deepcopy(self.wrapped_env.observation_space.low)
        num_domain_items = upper_domain_bound.shape[0]
        
        # TODO: There is no description to these features
        features = []
        for i in range(num_domain_items):
            lower = lower_domain_bound[i]
            upper = upper_domain_bound[i]

            if lower == -float('inf'):
                lower = np.finfo('float').min
            if upper == float('inf'):
                upper = np.finfo('float').max

            i_feature = RealFeature('feature_{}'.format(i), lower, upper)
            features.append(i_feature)

        self._observation_domain = ObservationDomain(features, num_agents=1)
        return {0: self._observation_domain}

    def _create_action_domains(self, config) -> Dict[Union[str, int], ActionDomain]:
        thigh_joint = DiscreteAction('thigh_joint', self.num_discrete_bins)
        leg_joint = DiscreteAction('leg_joint', self.num_discrete_bins)
        foot_joint = DiscreteAction('foot_joint', self.num_discrete_bins)
        self._action_domain = DiscreteActionDomain([thigh_joint, leg_joint, foot_joint], 1)

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
        pass
