from typing import Dict, Union, List, Tuple

import numpy as np

from domain.actions import DiscreteAction
from domain.observation import ObservationDomain
from domain import DiscreteActionDomain, ActionDomain
from domain.features import Feature, RealFeature
from common.properties import Properties
from config import Config
from config.config import ConfigItemDesc
from environment import Environment


class Drift(Environment):
    """
    Ultra-simple test control environment analogous to CartPole.
    """

    @classmethod
    def get_class_config(cls) -> List['ConfigItemDesc']:
        return []

    @classmethod
    def properties(cls) -> Properties:
        return Properties(use_discrete_action_space=True,
                          use_joint_observations=False,
                          use_agent_deletion=False,
                          use_agent_addition=False)

    def __init__(self, config: Config):
        # One agent, one class.
        self._agent_id_list = [0]
        self._agent_class_list = [0]
        self._agent_class_map = {0:0}
        self.time = 0

        super().__init__(config)

    # This env uses numpy for randomness, the numpy seed is set in the controller
    def set_initial_seed(self, seed: int):
        pass

    def get_seed_state(self):
        return []

    def set_seed_state(self, seed_state):
        pass

    def _reset_state(self, visualize: bool = False) -> np.ndarray:
        self.time = 0
        self.state = np.array([0.0, 0.0])
        return self.state

    def update(self, actions: Dict[int, np.ndarray]) -> Dict[int, int]:
        action = actions[0][0]
        # Terminate if gone too far out
        self.time += 1
        if np.abs(self.state[0]) > 10 or self.time > 200:
            self.done = True
        # Impulse velocity
        if action[0] == 1:
            self.state[1] += 0.1
        else:
            self.state[1] -= 0.1
        # Random accel
        self.state[1] += (np.random.rand() * 0.1) - 0.05
        # Apply velocity
        self.state[0] += self.state[1]
        return {0: 0 if self.done else 1}

    def observe(self, obs_groups: List[Tuple[int, ...]] = None) -> Dict[Union[int, Tuple[int, ...]], np.ndarray]:
        return {0: np.copy(self.state)}

    def _create_observation_domains(self, config) -> Dict[Union[str, int], ObservationDomain]:
        position = RealFeature('pos')
        velocity = RealFeature('vel')
        return {0: ObservationDomain([position, velocity], num_agents=1)}

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
        push = DiscreteAction('push', 2)
        return {0: DiscreteActionDomain([push], 1)}

    def visualize(self):
        pass
