"""
This version of BitFlip is compatible with a balanced hierarchy
The agent must select a bit in the best group of four (i.e closest to [0000])
and the best group of two in that group of four.
The penalty for the wrong flip is that the other group is reset

For example:
[0 1 0 1 1 1 1 0]: selecting action 4 would be the wrogn group of 4 and 2, so the state would be all ones
    since the other group of 4 would reset (1--> 4) and the other group of 2 would reset (7,8)

"""

from typing import Dict, List, Tuple, TYPE_CHECKING, Union
from environment.BitFlip import BitFlip
import numpy as np

if TYPE_CHECKING:
    from config import Config

class BitFlipPyramid(BitFlip):

    def __init__(self, config: 'Config'):
        super(BitFlipPyramid, self).__init__(config)

    def update(self, actions: Dict[int, np.ndarray]) -> Dict[int, int]:
        """
        :param actions:
        :return:
        """
        rewards = {}
        action_index = self.action_domain[self.agent_class].index_for_name('flip_actions')
        for agent in actions:
            action = actions[agent][action_index][0]

            # Action is bit to flip, so flip_0 flips 0th bit
            # bits are indexed right to left --> [6 5 4 3 2 1]
            i = self.num_bits - action

            reward = -1 * (2 ** i)
            bad_flip_2 = False
            bad_flip_4 = False

            # this is filled with the number of ones in each pair of points
            twos = np.zeros(int(self.num_bits/2))

            # this tracks the number of ones in each quartile
            fours = np.zeros(int(self.num_bits/4))

            # Check twos
            for i in range(4):
                if self.state_val[2*i] != self.state_val[2*i + 1]:
                    twos[i] = 1

                elif self.state_val[2*i] ==  1 and self.state_val[2*i + 1] == 1:
                    twos[i] = 2

            count = 0
            fours_ind = 0
            # Check fours
            for i in range(self.num_bits):
                fours[fours_ind] += 1 if self.state_val[i] == 1 else 0
                count += 1
                if count == 4:
                    count = 0
                    fours_ind += 1

            action_twos = int(action/2)
            action_fours = int(action/4)
            other_action_fours = (action_fours + 1) % 2

            # Error: picked the wrong set of twos in the subtask
            if twos[action_twos] == 2 and fours[action_fours] == 3:
                bad_flip_2 = True

            # Error: picked wrong group of four
            if fours[other_action_fours] != 0 and fours[action_fours] > fours[other_action_fours]:
                bad_flip_4 = True

            # Error: flipped a 0 when a 1 was available to flip
            if self.state_val[action] == 0:
                for i in range(self.num_bits):
                    self.state_val[i] = 1

            else:
                if bad_flip_4:
                    start = int(other_action_fours * 4)
                    end = start + 4
                    for i in range(start, end):
                        self.state_val[i] = 1

                if bad_flip_2:

                    # if a bad flip, reset entire group of four and do not flip intended action
                    # i.e [0, 1, 1, 1, 1, 1, 0, 0], action = 3 => [1, 1, 1, 1, 1, 1, 0, 0]
                    start = int(action_fours * 4)
                    end = start + 4
                    for i in range(start, end):
                        self.state_val[i] = 1

                elif not bad_flip_4:
                    self.state_val[action] = (self.state_val[action] + 1) % 2

            rewards[agent] = reward

        if np.count_nonzero(self.state_val) == 0:
            self.done = True
            self.eval = False

        return rewards