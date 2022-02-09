from typing import TYPE_CHECKING, List
import numpy as np

from callbacks import Callback, CallbackImpl
from common.utils import ensure_file_writable
from config.config import ConfigItemDesc

if TYPE_CHECKING:
    from config import Config
    from controller import MDPController


class SaveReward(Callback):
    """
    Save rewards per episode for each agent at the end of the run.
    """

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return [
            ConfigItemDesc(name='file_location',
                           check=lambda s: isinstance(s, str),
                           info='File to save reward to.'),
            ConfigItemDesc(name='discounted_reward',
                           check=lambda s: isinstance(s, bool),
                           info='Optional: Whether to save out discounted reward, default is false.',
                           optional=True, 
                           default=False)
        ]

    def __init__(self, controller: 'MDPController', config: 'Config'):
        self.file_location = ensure_file_writable(config.callbacks['SaveReward'].file_location)
        self.discounted_reward = config.callbacks['SaveReward'].discounted_reward

        with open(self.file_location, mode='w') as f:
            column_header = ['Policy Group, Episode, Experience, Reward']
            header = ','.join(column_header)
            f.write(header + '\n')

        super().__init__(controller, config)

    def _get_implement_flags(self):
        return CallbackImpl(after_episode=True)

    def after_episode(self):
        cumulative_results = {}
        for pg in self.controller.asys.policy_groups:
            pg_id = pg.pg_id
            if self.discounted_reward:
                total_reward = calculate_cumulative_r(self.controller.asys.algorithm.discount_factor,
                                                  pg.trajectory.rewards)
            else:
                total_reward = np.sum(pg.trajectory.rewards)
            cumulative_results[pg_id] = (len(pg.trajectory), total_reward)

        # Write results
        episode_num = self.controller.episode_num
        with open(self.file_location, mode='a') as f:
            for pg_id, pg_info in cumulative_results.items():
                experience, reward = pg_info
                row = [str(pg_id), str(episode_num), str(experience), str(reward)]
                row_str = ','.join(row)
                f.write(row_str + '\n')
    
def calculate_cumulative_r(gamma, rewards):
    curr_gamma = 1.0
    cumulative_r = 0.0
    for reward in rewards:
        cumulative_r += reward * curr_gamma
        curr_gamma *= gamma

    return cumulative_r
