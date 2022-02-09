import numpy as np
from typing import Dict, Any, TYPE_CHECKING, List

from callbacks import CallbackImpl
from common.utils import ensure_file_writable
from config.config import ConfigItemDesc
from config import checks
from . import Callback
if TYPE_CHECKING:
    from config import Config
    from controller import MDPController


class SaveBest(Callback):
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
            ConfigItemDesc(name='threshold',
                           check=checks.numeric,
                           info='A threshold minimum reward above which to save the agentsystem policies. '
                                'For no threshold, use "-Infinity".')
        ]

    def __init__(self, controller: 'MDPController', config: 'Config'):
        super().__init__(controller, config)
        # Map iteration number to a list of trajectories evaluated at that iteration.
        self.all_eval_reward_dicts = {}
        callback_config = config.callbacks['SaveBest']
        # (Folder) Location to save to
        self.file_location = ensure_file_writable(callback_config.file_location)
        # Threshold of reward for saving anything at all. Use "-Infinity" if you are unsure.
        self.best_mean = callback_config.threshold
        self.curr_episode = None
        self.rewards_at_curr_episode = []

    def _get_implement_flags(self):
        return CallbackImpl(after_update=True)

    def after_update(self):
        # TODO A lot of the functionality here is inconsistent and undefined. Work on bettering in future.
        # TODO refactor evaluation best over mean to Evaluate callback instead (in future)
        # Consider getting mean of N recent episodes or N evals instead.
        if not self.controller.episode_active():
            if self.curr_episode is None:
                # Since we analyze means at the first ending of a new episode
                self.curr_episode = self.controller.episode_num
            reward_dict = self.controller.curr_trajectory.get_agent_total_rewards()
            mean = np.mean([r for r in reward_dict.values()])
            if self.controller.flags.exploit:
                pass
            else:
                # A learning run. Those shouldn't really be repeated.
                if mean > self.best_mean:
                    print('Achieved best-so-far mean reward of [{}]. '
                          'Saving agentsystem to [{}]'.format(mean, self.file_location))
                    self.controller.asys.save(self.file_location)
                    self.best_mean = mean
            # In a different episode?
            if self.controller.episode_num == self.curr_episode:
                # Ended episode at existing iteration. Has to be from "evaluate" if under exploitation.
                return
            else:
                # First time we ended this new episode.
                pass

            self.curr_episode = self.controller.episode_num
