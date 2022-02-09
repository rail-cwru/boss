import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, TYPE_CHECKING, List

from callbacks import CallbackImpl
from config import checks
from config.config import ConfigItemDesc
from environment import HandcraftEnvironment
from . import Callback
if TYPE_CHECKING:
    from controller import MDPController
    from config import Config


class RunHandcrafted(Callback):
    """
    When triggered, runs the environment with handcrafted action selection.

    Useful to determine, for example, baselines or the quality of learning compared to
        some pre-set, non-policy based algorithm, etc.

    By default, NO OTHER CALLBACKS will be activated during these special runs.
    """

    # TODO refactor "TriggerOnEpisode" and "SpecialEpisode" somehow, or something

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
                           info='Whether or not to visualize evaluation episodes.')
        ]

    def __init__(self, controller: 'MDPController', config: 'Config'):
        # Map iteration number to a list of trajectories evaluated at that iteration.

        callback_config = config.callbacks['RunHandcrafted']
        # Number of datapoints to collect each eval (5 default)
        self.eval_num = callback_config.eval_num
        # Number of episodes between each eval
        self.eval_timestep = callback_config.timestep
        # Visualize the evaluation runs
        self.visualize = callback_config.visualize
        self.can_plot = 'PlotReward' in config.callbacks
        super().__init__(controller, config)
        self.lengths_at_times = {}       # episode -> (mean, stdv)
        self.all_eval_reward_dicts = {}  # agent -> episode -> (mean, stdv)

    def _get_implement_flags(self):
        return CallbackImpl(after_episode=True)

    def after_episode(self):
        episode_num = self.controller.episode_num
        if episode_num % self.eval_timestep == 0 or \
                episode_num == (self.controller.episodes - 1):
            # Do an eval of learned policy
            agent_rewards = {}
            lengths = []
            self.controller.flags.exploit = True
            self.controller.flags.learn = False
            print('Running handcrafted actions after iteration {}...'.format(episode_num))
            for _ in range(self.eval_num):
                self.controller.flags.visualize = self.visualize
                old_step = self.controller.step
                self.controller.step = self.replacement_step
                epi_traj = self.controller.run_episode()
                lengths.append(len(epi_traj))

                total_reward_dict = epi_traj.get_agent_total_rewards()
                for agent_id, reward in total_reward_dict.items():
                    if agent_id not in agent_rewards:
                        agent_rewards[agent_id] = [reward]
                    else:
                        agent_rewards[agent_id].append(reward)

                print('Total reward for agents in handcrafted: {}'.format(total_reward_dict))
                self.controller.step = old_step
                self.controller.flags.visualize = False
            for agent, rewards in agent_rewards.items():
                if agent not in self.all_eval_reward_dicts:
                    self.all_eval_reward_dicts[agent] = {}
                self.all_eval_reward_dicts[agent][episode_num] = rewards
            self.lengths_at_times[episode_num] = (np.mean(lengths), np.std(lengths))
            self._plot_data()

    def _plot_data(self):
        if not self.can_plot:
            return
        plt.figure('cumulative_reward')
        for agent, episode_rewards in self.all_eval_reward_dicts.items():
            data_x = []
            means = []
            stdvs = []
            scatter_x = []
            scatter_y = []
            for episode, rewards in episode_rewards.items():
                data_x.append(episode)
                means.append(np.mean(rewards))
                stdvs.append(np.std(rewards))
                for reward in rewards:
                    scatter_x.append(episode)
                    scatter_y.append(reward)
            plt.scatter(scatter_x, scatter_y, label='Agent {} (handcraft)'.format(agent), color='orange', s=0.5)
            plt.errorbar(data_x, means, stdvs, color='green', linewidth=0.7,
                         label='Agent {} (handcraft mean +/- stdev)'.format(agent))
        plt.legend()
        plt.draw()

        times = []
        l_means = []
        l_stdvs = []
        plt.figure('episode lengths')
        for episode_num, length in self.lengths_at_times.items():
            times.append(episode_num)
            l_means.append(np.mean(length))
            l_stdvs.append(np.std(length))
        plt.errorbar(times, l_means, l_stdvs, color='green', linewidth=0.7,
                     label='handcraft episode length +/- stdev')
        plt.legend()
        plt.draw()
        plt.pause(0.0001)

    def replacement_step(self):
        controller = self.controller
        assert isinstance(controller.env, HandcraftEnvironment)
        observation_request = controller.asys.observe_request()
        agents_observation = controller.env.observe(observation_request)
        if controller.flags.visualize:
            controller.env.visualize()
        agent_action_map = controller.env.handcraft_actions()
        agent_rewards = controller.env.update(agent_action_map)
        controller.curr_trajectory.append(controller.asys.agent_ids,
                                          agents_observation, agent_action_map, agent_rewards, controller.env.done)
        controller.episode_step += 1
