"""
Statistics module.
"""

from typing import TYPE_CHECKING, List
from callbacks import Callback, CallbackImpl
from matplotlib import pyplot as plt
import numpy as np

from config.config import ConfigItemDesc

if TYPE_CHECKING:
    from config import Config
    from controller import MDPController


class PlotReward(Callback):
    """
    Plots reward for standard RL experiments.

    Plots cumulative reward per episode for each agent at the end of the run.
    """

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return [
            ConfigItemDesc(name='no_plot',
                           check=lambda b: isinstance(b, bool),
                           info='Whether to skip plotting at end.',
                           optional=True, default=False)
        ]

    def __init__(self, controller: 'MDPController', config: 'Config'):
        self.no_plot = config.callbacks['PlotReward'].no_plot
        self.epi_lengths = []
        self.agent_total_rewards = {}
        self.agent_times = {}
        super().__init__(controller, config)

    def _get_implement_flags(self):
        return CallbackImpl(before_run=not self.no_plot, after_episode=True, finalize=True)

    def before_run(self):
        plt.ion()
        plt.figure('cumulative_reward').show()
        plt.figure('episode lengths').show()

    def after_episode(self):
        episode_traj = self.controller.episode_trajectories[-1]
        self.epi_lengths.append(len(episode_traj))

        time = self.controller.episode_num
        times = np.r_[1:time+1]
        total_rewards = episode_traj.get_agent_total_rewards()
        for agent, total_reward in total_rewards.items():
            if agent not in self.agent_total_rewards:
                self.agent_total_rewards[agent] = [total_reward]
                self.agent_times[agent] = [time]
            else:
                self.agent_total_rewards[agent].append(total_reward)
                self.agent_times[agent].append(time)

        if self.no_plot:
            return

        plt.figure('cumulative_reward').clear()
        plt.title('Cumulative reward for agents over episodes')
        for agent, total_rewards in self.agent_total_rewards.items():
            plt.plot(self.agent_times[agent], total_rewards, label="Agent {} (train)".format(agent))
        plt.legend()

        plt.figure('episode lengths').clear()
        plt.title('Episode lengths')
        plt.plot(times, self.epi_lengths, label="Episode lengths")
        plt.legend()
        plt.pause(0.02)

    def finalize(self):
        for agent, total_rewards in self.agent_total_rewards.items():
            if not self.no_plot:
                plt.plot(total_rewards, label="Agent {} (train)".format(agent))
            print('Non-Eval Rewards:', agent, total_rewards)
        print('Episode Lengths: ', self.epi_lengths)

        if not self.no_plot:
            plt.ioff()
            plt.show()

        plt.close('all')

        return {'agent_total_rewards': self.agent_total_rewards, 'lengths': self.epi_lengths}
