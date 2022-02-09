import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, TYPE_CHECKING, List

from callbacks import CallbackImpl
from common.utils import ensure_file_writable
from config import checks
from config.config import ConfigItemDesc
from . import Callback
if TYPE_CHECKING:
    from controller import MDPController
    from config import Config

from common.trajectory import TrajectoryCollection


class Evaluate(Callback):
    """
    Evaluate the agentsystem (deterministic argmax action selection if available) periodically
        and add to the cumulative reward plot if plotting cumulative reward at end.

    Will plot cumulative episode reward at episode num alongside mean +/- stdev errorbars.

    If visualizing, the evaluation rewards will also be shown at the end of episode.

    The Evaluate callback runs ADDITIONAL episodes that DO NOT learn.

    VISUALIZE WITH DISCRETION
    """

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return [
            ConfigItemDesc(name='eval_num',
                           check=checks.positive_integer,
                           info='Number of times to evaluate when triggered'),
            ConfigItemDesc(name='timestep',
                           check=checks.positive_integer,
                           info='Interval of episodes after which to run this callback.'),
            ConfigItemDesc(name='visualize',
                           check=lambda b: isinstance(b, bool),
                           info='Whether or not to visualize evaluation episodes.'),
            ConfigItemDesc(name='save_best_mean',
                           check=lambda s: isinstance(s, str),
                           info='Optional: File to save policy that yields highest mean cumulative reward over runs.',
                           optional=True, default=''),
            ConfigItemDesc(name='output_reward_file',
                           check=lambda s: isinstance(s, str),
                           info='Optional: File to store evaluation cumulative rewards in.',
                           optional=True, default=''),
            ConfigItemDesc(name='output_trajectory_file',
                           check=lambda s: isinstance(s, str),
                           info='Optional: File to store evaluation trajectories in.',
                           optional=True, default=''),
            ConfigItemDesc(name='pickle',
                           check=lambda b: isinstance(b, bool),
                           info='Whether or not to save the eval reults in a pickle.',
                           optional=True, default=True),
        ]

    def __init__(self, controller: 'MDPController', config: 'Config'):
        # Map iteration number to a list of trajectories evaluated at that iteration.
        self.all_reward_dicts = {}
        callback_config = config.callbacks['Evaluate']

        self.plot_at_end = False

        self.pickle = callback_config.pickle
        self.pickle = True

        # Number of datapoints to collect each eval (5 default)
        self.eval_num = callback_config.eval_num
        # Number of episodes between each eval
        self.eval_timestep = callback_config.timestep
        # Visualize the evaluation runs
        self.visualize = callback_config.visualize

        self.output_reward_file = ensure_file_writable(callback_config.output_reward_file)
        self.output_trajectory_file = ensure_file_writable(callback_config.output_trajectory_file)
        self.save_best_mean = ensure_file_writable(callback_config.save_best_mean)

        self.trajectories = {}
        self.best_exploit_mean = float('-inf')
        self.plot_at_end = 'PlotReward' in config.callbacks
        self.all_exploit_reward_dicts = {}

        if self.output_reward_file:
            with open(self.output_reward_file, 'w') as reward_file:
                reward_header = ['After Episode Number', 'Policy Group', 'Avg Eval Reward', 'Std Dev Eval Reward']
                reward_header_str = ','.join(reward_header)
                reward_file.write(reward_header_str + '\n')

        self.agent_rew_list = {}
        self.agent_max_list = {}
        self.all_averages = []

        super().__init__(controller, config)

    def _get_implement_flags(self):
        return CallbackImpl(after_episode=True, after_run=True)

    def run_episodes(self):
        trajs = {}
        reward_dicts = []
        mean_rewards = []
        max_rew = -1 * float("inf")
        #print('eval')
        # plt.ion()
        # plt.figure('eval reward').show()

        for i in range(self.eval_num):
            self.controller.flags.visualize = self.visualize
            t = self.controller.run_episode()

            total_reward_dict = {}
            traj_dict = {}

            if hasattr(self.controller.asys, 'hierarchy'):
                for pg_id in t.get_agent_trajectories().keys():
                    trajectory = t.get_agent_trajectories()[pg_id]
                    traj_dict[pg_id] = trajectory
                    total_reward_dict[pg_id] = np.sum(trajectory.rewards)
            else:
                # todo why zero?
                for pg_id, pg in enumerate(self.controller.asys.policy_groups):
                    traj_dict[pg_id] = pg.trajectory
                    total_reward_dict[pg_id] = np.sum(pg.trajectory.rewards)

            if self.output_trajectory_file:
                trajs[i] = traj_dict
            
            mean_rewards.append(np.mean([r for r in total_reward_dict.values()]))
            self.best_exploit_mean = max(self.best_exploit_mean, mean_rewards[-1])
            reward_dicts.append(total_reward_dict)

            self.controller.flags.visualize = False
        self.all_exploit_reward_dicts[self.controller.episode_num] = reward_dicts
        self.all_averages.append(np.mean(mean_rewards))
        # print('Max Reward:', max(mean_rewards))
        return reward_dicts, mean_rewards, trajs

    def after_episode(self):
        episode_num = self.controller.episode_num
        if episode_num % self.eval_timestep == 0 or episode_num == (self.controller.episodes - 1):
            self.controller.flags.exploit = True
            self.controller.flags.learn = False
            #print('Evaluating learned policy after iteration {}...'.format(episode_num))

            # Set exploit episode length
            old_len = self.controller.episode_max_length
            self.controller.episode_max_length = min(self.controller.eval_max_length, old_len)

            # Run pure exploitive episodes
            reward_dicts, mean_rewards, trajs = self.run_episodes()

            # Reset episode length
            self.controller.episode_max_length = old_len

            mean_over_eval = np.mean(mean_rewards)
            #print('Average Reward with Exploit: {}'.format(mean_over_eval))

            if self.save_best_mean and mean_over_eval > self.best_exploit_mean:
                target = self.save_best_mean + '.npz'
                #print('Achieved best-so-far mean reward of [{}] with exploitation. '
                #      'Saving agentsystem to [{}]'.format(mean_over_eval, target))
                self.controller.asys.save(target)
                self.best_exploit_mean = mean_over_eval

            self.all_reward_dicts[episode_num] = reward_dicts
            if self.output_trajectory_file:
                self.trajectories[episode_num] = trajs

            # plt.figure('eval reward').clear()
            # plt.title('Eval Rewards')
            for agent_id in reward_dicts[0].keys():
                if agent_id not in self.agent_rew_list:
                    self.agent_rew_list[agent_id] = []
                if agent_id not in self.agent_max_list:
                    self.agent_max_list[agent_id] = []
                    self.agent_max_list[agent_id].append(mean_over_eval)
                else:
                    self.agent_max_list[agent_id].append(max(self.agent_max_list[agent_id][-1], mean_over_eval))
                self.agent_rew_list[agent_id].append(mean_over_eval)

                time_step = range(0, episode_num, self.eval_timestep)
                if len(time_step) == len(self.agent_rew_list[agent_id]):
                    plt.plot(time_step, self.agent_rew_list[agent_id])
                    plt.plot(time_step, self.agent_max_list[agent_id])

            # plt.show()

            # Write file as we go
            if self.output_reward_file:
                with open(self.output_reward_file, 'a') as reward_file:
                    reward_dict = self.all_reward_dicts[episode_num]
                    for pg_id in reward_dict[0].keys():
                        eval_rewards = [reward[pg_id] for reward in reward_dict]
                        row = [episode_num, pg_id, np.mean(eval_rewards), np.std(eval_rewards)]
                        row = [str(x) for x in row]
                        row_str = ','.join(row)
                        reward_file.write(row_str + '\n')

            #print(self.agent_rew_list[0])

    def after_run(self):

        # print('Eval:')
        # print(self.all_averages)

        # arr = pickle.load(open('eval', 'rb'))
        # arr.append(self.all_averages)
        # pickle.dump(arr, open('eval', 'wb'))

        if self.output_trajectory_file:
            # TODO accumulating trajs in memory is VERY COSTLY - find way to save to disk @ each time
            pickle.dump(self.trajectories, file=open(self.output_reward_file, 'wb+'))

        if not self.plot_at_end:
            return

        #plt.figure('cumulative_reward')
        rewards_at_times = {}  # a -> t -> r
        agent_total_rewards: Dict[int, Any] = {}  # a -> (t, r)
        for episode_num, reward_dicts in self.all_exploit_reward_dicts.items():
            for total_reward_dict in reward_dicts:
                for agent, total_reward in total_reward_dict.items():
                    if agent not in agent_total_rewards:
                        agent_total_rewards[agent] = []
                    if agent not in rewards_at_times:
                        rewards_at_times[agent] = {}
                    if episode_num not in rewards_at_times[agent]:
                        rewards_at_times[agent][episode_num] = []

                    agent_total_rewards[agent].append((episode_num, total_reward))
                    rewards_at_times[agent][episode_num].append(total_reward)

        for agent, rewards_at_time in rewards_at_times.items():
            x = []
            y = []
            for x_, y_ in agent_total_rewards[agent]:
                x.append(x_)
                y.append(y_)
            plt.scatter(x, y, label='Agent {} (eval)'.format(agent), color='orange', s=0.5)
            times = []
            means = []
            stdvs = []
            for episode_num, rewards in rewards_at_time.items():
                times.append(episode_num)
                means.append(np.mean(rewards))
                stdvs.append(np.std(rewards))
            plt.errorbar(times, means, stdvs, color='red', linewidth=0.5,
                         label='Agent {} (eval mean +/- stdev)'.format(agent))

