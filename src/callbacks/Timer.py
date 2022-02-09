import time         # Switch for a better time library
from typing import Dict, Any, TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt

from callbacks import CallbackImpl
from . import Callback
if TYPE_CHECKING:
    from controller import MDPController
    from config import Config


class Timer(Callback):
    """
    Time various portions of the controller's runtime loop.

    Since episodes can terminate earlier than the episode time limit, we instead calculate running averages over
        the duration of the episode of the actual observe, update, and learn steps.

    Avoids timing when the controller is visualizing or not learning.
    """

    def __init__(self, controller: 'MDPController', config: 'Config'):
        super().__init__(controller, config)
        self.t = self.reset_episode()
        self.episode_t = {k: [] for k in self.t.keys()}
        self.step_clock = 0.0   # The last call at the beginning of the episode step
        self.last_clock = 0.0   # The immediate last call

    def reset_episode(self):
        return {
            'observe': [],
            'action': [],
            'update': [],
            'learn': [],
            'step': []
        }

    def _get_implement_flags(self):
        return CallbackImpl(before_episode=True,
                            before_observe=True,
                            on_observe=True,
                            on_action=True,
                            on_update=True,
                            after_update=True,
                            after_episode=True,
                            after_run=True,
                            finalize=True)

    def lap(self):
        t = time.clock()
        dt = t - self.last_clock
        self.last_clock = t
        return dt

    def before_episode(self):
       self.t = self.reset_episode()

    def before_observe(self):
        self.step_clock = self.last_clock = time.clock()

    def on_observe(self, agents_observation: Dict[Any, np.ndarray]):
        self.t['observe'].append(self.lap())
        return agents_observation

    def on_action(self,
                  agents_observation: Dict[Any, np.ndarray],
                  agent_action_map: Dict[Any, np.ndarray]):
        self.t['action'].append(self.lap())
        return agents_observation, agent_action_map

    def on_update(self,
                  agents_observation: Dict[Any, np.ndarray],
                  agent_action_map: Dict[Any, np.ndarray],
                  agent_rewards: Dict[Any, float]):
        self.t['update'].append(self.lap())
        return agents_observation, agent_action_map, agent_rewards

    def after_update(self):
        self.t['learn'].append(self.lap())
        self.t['step'].append(time.clock() - self.step_clock)

    def after_episode(self):
        flags = self.controller.flags
        epi_num = self.controller.episode_num
        if flags.learn and not flags.exploit and not flags.visualize:
            for k, times in self.t.items():
                self.episode_t[k].append((epi_num, np.mean(times), np.std(times)))

    def after_run(self):
        plt.figure('Timing')
        for k, timedata in self.episode_t.items():
            t, mean, std = zip(*timedata)
            if k != 'step':
                fig = plt.figure('Timing')
            else:
                fig = plt.figure('Episode Step Timing')
            plt.errorbar(t, mean, std, linewidth=0.5,
                         label=k, figure=fig)
        # TODO fix plot figure id collisions & related insanity

    def finalize(self):
        plt.figure('Timing')
        plt.title('Timing')
        plt.ylabel('Seconds')
        plt.legend()

        plt.figure('Episode Step Timing')
        plt.title('Episode Step Timing')

        # plt.show()
