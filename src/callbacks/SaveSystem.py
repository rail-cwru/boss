import os

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, TYPE_CHECKING, List

from callbacks import CallbackImpl
from common.utils import ensure_file_writable
from config.config import ConfigItemDesc
from config import checks
from . import Callback
if TYPE_CHECKING:
    from config import Config
    from controller import MDPController


class SaveSystem(Callback):
    """
    Saves the policy that obtained the greatest mean cumulative reward over n runs during evaluation.

    Recommended to use with Evaluate to save best policies under exploitation.
    """

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return [
            ConfigItemDesc(name='file_location',
                           check=lambda s: isinstance(s, str),
                           info='Location of policy npz file to save as'),
            ConfigItemDesc(name='timestep',
                           check=checks.numeric,
                           info='Episode intverval after which to save the agentsystem.')
        ]

    def __init__(self, controller: 'MDPController', config: 'Config'):
        super().__init__(controller, config)
        # Map iteration number to a list of trajectories evaluated at that iteration.
        self.all_eval_reward_dicts = {}
        callback_config = config.callbacks['SaveSystem']
        # (Folder) Location to save to
        self.file_location = ensure_file_writable(callback_config.file_location)
        # Threshold of reward for saving anything at all. Use "-Infinity" if you are unsure.
        self.eval_timestep = callback_config.timestep
        self.leftpad_zeros = len(str(self.controller.episodes))
        self.episode_num = None

    def _get_implement_flags(self):
        return CallbackImpl(after_update=True)

    def _get_target(self):
        return self.file_location + '_{{:0{}}}'.format(self.leftpad_zeros).format(self.controller.episode_num)

    def after_update(self):
        if not self.controller.episode_active():
            # Only save the first time we end a new episode.
            episode_num = self.controller.episode_num
            final_episode = episode_num == (self.controller.episodes - 1)
            if episode_num == self.episode_num and not final_episode:
                return
            if episode_num % self.eval_timestep == 0 or final_episode:
                print('Agentsystem saved to [{}]'.format(self._get_target()))
                self.controller.asys.save(self._get_target())
            self.episode_num = episode_num

