"""
BitFlip Domain: Flip a sequence of bits from left to right until the sequence is all zeros
The penalty for not flipping left to right is resetting all bits to the left
reward is -2^(num_bits - flipped bit) so that the first bit has the largest negative reward

"""
import random
from typing import Dict, List, Tuple, TYPE_CHECKING, Union

from environment.BitFlip import BitFlip

import numpy as np

if TYPE_CHECKING:
    from config import Config
# 1, 2, 3, 2, 3, 3, 4, 5, 3, 3, 3, 3, 4, 2, 3, 3 = 47 = 2.94
start_set = [
              [0, 0, 1, 1, 1, 0, 0, 0],
              [0, 1, 0, 0, 1, 0, 0, 0],
              [1, 0, 0, 0, 1, 0, 0, 0],
              [1, 1, 1, 1, 1, 0, 0, 0],
              [1, 1, 1, 1, 1, 0, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1],
              [0, 1, 1, 0, 1, 0, 0, 0],
              [0, 1, 1, 0, 0, 1, 1, 0],
              [1, 0, 0, 1, 1, 1, 0, 1],
              [0, 0, 1, 0, 1, 1, 0, 1],
              [0, 0, 0, 1, 0, 1, 0, 1],
              [1, 1, 0, 0, 1, 1, 0, 1],
              [1, 1, 1, 1, 0, 1, 0, 1],
              [1, 1, 0, 0, 1, 1, 1, 0],
              [1, 0, 1, 0, 1, 1, 0, 0],
              [1, 1, 0, 1, 1, 0, 1, 1]
            ]


class BitFlipThree(BitFlip):

    def __init__(self, config: 'Config'):
        super(BitFlipThree, self).__init__(config)

    def _set_environment_init_state(self) -> np.ndarray:
        """
        Initialize the state with the config.
        :return: Initial state
        """
        self.state_val = random.choice(start_set)[:]
        return self.state_val

    def _reset_state(self, visualize: bool = False) -> np.ndarray:
        """
        Initialize the state with the config.
        :return: Initial state
        """
        if not self.eval:
            self.state_val = random.choice(start_set)[:]
        return self.state_val

    def update(self, actions: Dict[int, np.ndarray]) -> Dict[int, int]:
        """
        :param actions:
        :return:
        """
        rewards = {}
        action_index = self.action_domain[self.agent_class].index_for_name('flip_actions')
        for agent in actions:

            action = actions[agent][action_index][0]

            # Action is bit to flip, so flip_0 flips 0th bit (with adjacent as well)
            # bits are indexed right to left --> [6 5 4 3 2 1]
            i = self.num_bits - action
            reward = -1

            # Check if there is a bit to the left, will NOT wrap around to the end
            if action > 0:
                action_left = action - 1
                self.state_val[action_left] = (self.state_val[action_left] + 1) % 2

            # Check if there is a bit to the right, will NOT wrap around to the end
            if action < self.num_bits-1:
                action_right = action + 1
                self.state_val[action_right] = (self.state_val[action_right] + 1) % 2

            self.state_val[action] = (self.state_val[action] + 1) % 2
            rewards[agent] = reward

        if np.count_nonzero(self.state_val) == 0:
            self.done = True
            self.eval = False

        return rewards
