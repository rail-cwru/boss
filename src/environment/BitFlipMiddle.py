
import random
from typing import Dict, List, Tuple, TYPE_CHECKING, Union
from environment.BitFlip import BitFlip
import numpy as np

if TYPE_CHECKING:
    from config import Config


class BitFlipMiddle(BitFlip):
    """
    BitFlip Domain: Flip a sequence of bits from from the middle out until the sequence is all zeros
    The penalty for not flipping middle out is resetting all bits in the middle of the wrong action
    reward is -2^(num_bits - flipped bit) so that the first bit has the largest negative reward

    """

    def __init__(self, config: 'Config'):
        super(BitFlipMiddle, self).__init__(config)
        # self.use_penalty = True
        print("Use Penalty: ", self.use_penalty)
        print("Small Penalty", self.small_penalty)

    def update(self, actions: Dict[int, np.ndarray]) -> Dict[int, int]:
        """
        :param actions:
        :return:
        """
        rewards = {}
        action_index = self.action_domain[self.agent_class].index_for_name('flip_actions')
        for agent in actions:
            action = actions[agent][action_index][0]
            max_rew = -1 * self.num_bits/2
            action_pos = action + 1 if action < self.num_bits/2 else action
            reward = -1 * (2 ** abs(max_rew + abs(self.num_bits/2 - action_pos)))
            bad_flip = False

            even = True if self.num_bits % 2 == 0 else False
            middle = self.num_bits/2
            if even:
                middle2 = middle-1
                distance_from_middle = int(min(abs(action - middle), abs(action - middle2)))
                inside = []
                for i in range(distance_from_middle):
                    inside.append(int(middle + i))
                    inside.append(int(middle2 - i))
            else:
                raise ValueError('Odd Bit length not yet supported')

            for ins in inside:
                if self.state_val[ins] == 1:
                    bad_flip = True
                    break
            if bad_flip:
                for ins in inside:
                    self.state_val[ins] = 1
                self.state_val[action] = 1
            # Flip bit
            else:
                self.state_val[action] = (self.state_val[action] + 1) % 2
            rewards[agent] = reward
        if np.count_nonzero(self.state_val) == 0:
            self.done = True
            self.eval = False
        return rewards
