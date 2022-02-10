"""
The standard provided entrypoint to run the framework with a specified config file.
Takes two arguments:
    1) Path to the config file
    2) A boolean indicating if the analysis is to be performed offline
        This value defaults to False, indicating online analysis

    Results are saved in both .txt and .pkl files as well as printed to standard out
"""
import os
os.environ['PYTHONHASHSEED'] = str('0')
import json
import argparse
import numpy as np
import time
import datetime
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pickle as pkl

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Uncomment to use CPU only

# TODO have controller imported via config name to finally unify classes
# from controller import MDPController
# from controller.offline_controller import OfflineController
# from config import Config

class ResultsManager():


    def __init__(self, offline):
        # These arrays are used to store results
        self.rew_arr = []
        self.eval_arr = []
        self.len_arr = []
        self.trajectories = []
        self.distribution_arr = []
        self.derived_arr = []
        self.rt_arr = []
        self.times = []
        self.cummax_arr = []

        self.kl_divergences = []
        self.kl_lens = []
        self.entropies = []
        self.target_entropy = []
        self.novel_sa = []
        self.current_date = datetime.datetime.now()

        self.boss_rew_dict = {}
        self.boss_ucb_dict = {}
        self.boss_cum_rew_dict = {}
        self.boss_cum_ucb_dict = {}
        self.offline = offline
        self.boss_schedule_list = []


        self.dir_name = None


    def after_run(self, config, controller, save_name, is_boss, collect_novel, iteration):
        # This section looks at saving and displaying the results
        if 'DispResults' in config.callbacks:
            self._display_results(controller)

        if 'Evaluate' in config.callbacks or 'EvaluateBitflip' in config.callbacks:
            self.append_all_averages(controller)

        if self.offline:
            self.append_offline_results(controller, config.kl_divergence)


        if controller.save_traj:
            self.append_trajectories(controller)

        if collect_novel:
            self.append(controller.novel_sa_visits)

        if 'DispResults' in config.callbacks:
            self.display_results()

        self._display_offline()

        if is_boss:
            if iteration == 0:
                self.initialize_boss_dict(controller)

            self.append_boss(controller)
            self.plot_boss(controller, iteration)

        # if self.offline:
        ch_dir = self.save_all_lists(controller, save_name, is_boss, collect_novel)

        if ch_dir:
            os.chdir('..')

        if 'Evaluate' in config.callbacks or 'EvaluateBitflip' in config.callbacks:
            self.display_eval()

        if self.offline and hasattr(config, 'display_distribution') and config.display_distribution:
            self._display_distribution(controller)

    def set_dir_name(self, dir_name):
        self.dir_name = dir_name

    def append_offline_results(self, controller, kl=False):
        if self.offline:

            self.eval_arr.append(controller.rewards)
            self.len_arr.append(controller.lens)
            self.derived_arr.append(controller.derived_samples)
            self.rt_arr.append(controller.rt)

            if kl:
                self.kl_divergences.append(controller.kl_divergences)
                self.kl_lens.append(controller.kl_lens)
                self.entropies.append(controller.entropies)
                target_entropy = controller.target_entropy
                self._display_kl(target_entropy)

    def append_all_averages(self, controller):
        self.eval_arr.append(controller.callbacks[0].all_averages)

    def append_times(self, time):
        self.times.append(time)

    def append_trajectories(self, controller):
        # Save Trajectory (i.e primitive actions taken)
        self.trajectories.append(controller.all_traj)
        prim_actions = controller.env.action_domain[0].full_range
        traj_dist = np.zeros(prim_actions)
        for i in controller.all_traj:
            traj_dist[i] += 1

        traj_dist = traj_dist / sum(traj_dist)
        if hasattr(controller.env, "hierarchy"):
            for k, v in controller.env.hierarchy.primitive_action_map.items():
                print(k, traj_dist[v])
        else:
            print(traj_dist)

    def append_novel_sa(self, controller):
        self.novel_sa.append(controller.novel_sa_visits)

    def display_results(self):
        print('Episode Rewards')
        for i in self.rew_arr:
            print(i)

        print('Episode Lengths')
        for i in self.len_arr:
            print(i)

    def display_eval(self):
        print('Eval')
        for i in self.eval_arr:
            print(i)

    def initialize_boss_dict(self, controller):
        for k, v in controller.sampler_reward_dict.items():
            self.boss_rew_dict[k] = []
            self.boss_ucb_dict[k] = []
            self.boss_cum_rew_dict[k] = []
            self.boss_cum_ucb_dict[k] = []

    def append_boss(self, controller):
        for k, v in controller.sampler_reward_dict.items():

            if k is not 'BOSS':
                # print(k, 'Rewards', v)
                self.boss_rew_dict[k].append(v)
                self.boss_ucb_dict[k].append(controller.sampler_ucb_dict[k])
                # print(k, 'UCB:', controller.sampler_ucb_dict[k])
                print('UCB', k, ':')
                for i in self.boss_ucb_dict[k]:
                    cum_ucb = [x for x in np.maximum.accumulate(i)]
                    self.boss_cum_ucb_dict[k].append(cum_ucb)
                    print(cum_ucb)
                    # print(cum_ucb)
                print('Rewards', k, ':')
                for i in self.boss_rew_dict[k]:
                    cum_rew = [x for x in np.maximum.accumulate(i)]
                    self.boss_cum_rew_dict[k].append(cum_rew)
                    print(cum_rew)

        self.boss_schedule_list.append(controller.sampler_schedule_arr)
        print('Schedule:')
        for i in self.boss_schedule_list:
            print(i)


    def plot_boss(self, controller, iteration):
        linestyle_tuple = [
            ('loosely dotted', (0, (1, 10))),
            ('dotted', (0, (1, 1))),
            ('dashed', (0, (5, 5))),
            ('loosely dashdotted', (0, (3, 10, 1, 10))),

            ('loosely dashed', (0, (5, 10))),
            ('densely dotted', (0, (1, 1))),
            ('densely dashed', (0, (5, 1))),

            ('dashdotted', (0, (3, 5, 1, 5))),
            ('densely dashdotted', (0, (3, 1, 1, 1))),

            ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
            ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
            ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

        count = 0
        for k, v in controller.sampler_reward_dict.items():
            if k is not "BOSS":
                plt.plot(self.len_arr[-1], self.boss_ucb_dict[k][-1], label=k,
                         linestyle=linestyle_tuple[count][-1])
                count += 1
        plt.legend()
        plt.title('BOSS UCB Vs. Samplers Collected')

        # plt.xlim(0, np.mean(len_arr, axis = 0)[-1])
        plt.xlabel('Samples Collected (all)')
        plt.ylabel('UCB')
        os.chdir(self.dir_name)
        plt.savefig('UCB_' + str(iteration) + '.png')

        os.chdir('..')
        self.boss_rew_dict["BOSS"].append(controller.sampler_reward_dict['BOSS'])

        print('Rewards', "Boss", ':')
        for i in self.boss_rew_dict["BOSS"]:
            cum_ucb = [x for x in np.maximum.accumulate(i)]
            self.boss_cum_rew_dict["BOSS"].append(cum_ucb)
            print(cum_ucb)

        count = 0

        plt.clf()
        for k, v in controller.sampler_reward_dict.items():
            plt.plot(self.len_arr[-1], self.boss_cum_rew_dict[k][-1], label=k,
                     linestyle=linestyle_tuple[count][-1])
            count += 1
        plt.legend()
        plt.title('BOSS Rewards Vs. Samples Collected:' + str(iteration))

        # plt.xlim(0, np.mean(len_arr, axis = 0)[-1])
        plt.xlabel('Samples Collected (all)')
        plt.ylabel('Reward')
        os.chdir(self.dir_name)
        plt.savefig('Rewards_' + str(iteration) + '.png')

        controller._write_sample_distribution(iteration)

        if controller.sampler.check_dist:
            for k, v in controller.sampler.sampler_dist_dict.items():
                x = []
                y = []
                for k_1, v_1 in v.items():
                    x.append(k_1)
                    y.append(v_1)
                y = [i / sum(y) for i in y]

                plt.clf()
                plt.bar(x, y)
                plt.ylabel('Percentage of action selection')
                plt.xlabel('Action')
                plt.title('Action Frequency: ' + k)
                plt.savefig('action_frequency_' + k + '_' + str(iteration))

        os.chdir('..')

        plt.clf()
        count = 0
        for k, v in controller.sampler_reward_dict.items():
            if k is not "BOSS":
                plt.plot(np.mean(self.len_arr, axis=0), np.mean(self.boss_ucb_dict[k], axis=0), label=k,
                         linestyle=linestyle_tuple[count][-1])
                count += 1
        plt.legend()
        plt.title('BOSS UCB Vs. Samplers Collected')

        # plt.xlim(0, np.mean(len_arr, axis = 0)[-1])
        plt.xlabel('Samples Collected (all)')
        plt.ylabel('UCB')
        plt.savefig('UCB.png')

        ## Plot each sampler:
        for k, v in controller.sampler_reward_dict.items():
            if k is not "BOSS":
                plt.clf()

                plt.plot(np.mean(self.len_arr, axis=0), np.mean(self.boss_ucb_dict[k], axis=0), label='UCB',
                         linestyle=linestyle_tuple[0][-1])

                plt.plot(np.mean(self.len_arr, axis=0), np.mean(self.boss_rew_dict[k], axis=0), label="REW",
                         linestyle=linestyle_tuple[1][-1])
                plt.title('UCB vs. Rew: ' + k)
                plt.xlabel('Samples Collected')
                plt.ylabel('Reward')
                plt.legend()
                plt.savefig(k + '.png')

        plt.clf()

        count = 0
        for k, v in controller.sampler_reward_dict.items():
            plt.plot(np.mean(self.len_arr, axis=0), np.mean(self.boss_cum_rew_dict[k], axis=0), label=k,
                     linestyle=linestyle_tuple[count][-1])
            count += 1
        plt.legend()
        plt.title('BOSS Rewards Vs. Samples Collected')

        # plt.xlim(0, np.mean(len_arr, axis = 0)[-1])
        plt.xlabel('Samples Collected (all)')
        plt.ylabel('Reward')
        plt.savefig('Rewards.png')

        plt.clf()

        count = 0
        for k, v in controller.sampler_reward_dict.items():
            if k is not "BOSS":
                plt.plot(np.mean(self.len_arr, axis=0), np.mean(self.boss_ucb_dict[k], axis=0), label=k,
                         linestyle=linestyle_tuple[count][-1])
                count += 1
        plt.legend()
        plt.title('BOSS UCB Vs. Samplers Collected')

        # plt.xlim(0, np.mean(len_arr, axis = 0)[-1])
        plt.xlabel('Samples Collected (all)')
        plt.ylabel('UCB')
        plt.savefig('UCB.png')

        ## Plot each sampler:
        for k, v in controller.sampler_reward_dict.items():
            if k is not "BOSS":
                plt.clf()

                plt.plot(np.mean(self.len_arr, axis=0), np.mean(self.boss_ucb_dict[k], axis=0), label='UCB',
                         linestyle=linestyle_tuple[0][-1])

                plt.plot(np.mean(self.len_arr, axis=0), np.mean(self.boss_rew_dict[k], axis=0), label="REW",
                         linestyle=linestyle_tuple[1][-1])
                plt.title('UCB vs. Rew: ' + k)
                plt.xlabel('Samples Collected')
                plt.ylabel('Reward')
                plt.legend()
                plt.savefig(k + '.png')

        plt.clf()

        ## plot schedule!

        sampler_dict = {}
        sampler_list = ['HUF', 'Polled', 'TDF_1', 'TDF_2', 'TDF_3', 'TDF_4', 'BUF_1', 'BUF_2',
                        'BUF_3', 'BUF_4', "BUF_5", "BUF_6", "BUF_7", 'TDF_5', "TDF_6"]

        for i in sampler_list:
            sampler_dict[i] = np.zeros(len(np.mean(self.len_arr, axis=0)))

        plot_set = set()
        l_c = 0
        for i in range(len(np.mean(self.len_arr, axis=0))):
            count_dict = {}

            l_c = 0
            for j in self.boss_schedule_list:
                if j[i] not in count_dict:
                    count_dict[j[i]] = 1
                else:
                    count_dict[j[i]] += 1
                l_c += 1

            for c in count_dict:
                sampler_dict[c][i] = count_dict[c] / l_c
                plot_set.add(c)

        plt.clf()
        count = 0
        for pl in plot_set:
            plt.plot(np.mean(self.len_arr, axis=0), sampler_dict[pl], label=pl)

            # linestyle=linestyle_tuple[count][-1])
            count += 1

        plt.legend()
        plt.title("Schedule")
        plt.xlabel('Total Samples Collected')
        plt.ylabel('Percentage of times sampler was chosen')
        plt.savefig("Schedule")

        window = 3
        plt.clf()
        count = 0
        for pl in plot_set:
            plt.plot(np.mean(self.len_arr, axis=0)[:-1 * (window - 1)],
                     self.moving_average(sampler_dict[pl], window),
                     label=pl)

            # linestyle=linestyle_tuple[count][-1])
            count += 1

        plt.legend()
        plt.title("Schedule" ' MA')
        plt.xlabel('Total Samples Collected')
        plt.ylabel('Percentage of times sampler was chosen')
        plt.savefig("Schedule" + '_ma')

    def _display_offline(self):

        """
        prints results to stdout for an offline analysis
        """
        if self.offline:
            print('Episode Lengths')
            # save_list(len_arr, "len_", args.name)
            for i in self.len_arr:
                print(i)

            print('Eval')
            for i in self.eval_arr:
                print(i)

            print('Cummax')

            for i in self.eval_arr:
                print([x for x in np.maximum.accumulate(i)])
                self.cummax_arr.append([x for x in np.maximum.accumulate(i)])

            print('Derived Samples:')
            for i in self.derived_arr:
                print(i)

            print("times:")
            for i in self.times:
                print(i)

    def _display_kl(self, target_entropy):
        """
        Prints KL divergence data to stdout
        :return:
        """
        print('KL Divergences: ')
        for i in self.kl_divergences:
            print(i)

        print('KL Lens:')
        for i in self.kl_lens:
            print(i)

        print('Entropy:')
        for i in self.entropies:
            print(i)

        print('Target Entropy:', target_entropy)

    def _display_distribution(self, controller):

        """
        Method used to save the distribution of primitive actions as a list
        """
        self.distribution_arr.append([controller.sampler.get_distribution()])
        target_list = self.distribution_arr
        print(target_list)
        with open('target_distribution.list', 'wb') as target_dist_file:
            pkl.dump(target_list, target_dist_file)

    def save_list(self, list, name, args_name):

        """
        Saves the results of the analysis as a pkl
        :param list: List to save. Can be any dimension
        :param name: prefix of saved file name
        :param args_name: suffix of saved file name
        """
        complex_name = name + args_name + "_master" + ".txt"
        pkl_name = name + args_name + "_master" + ".pkl"
        f = open(pkl_name, 'wb')

        pkl.dump(list, f)
        with open(complex_name, "w") as txt_file:
            for ind, i in enumerate(list):
                txt_file.write("[" + str(i)[1:-1] + "]\n")

    def save_all_lists(self, controller, save_name, is_boss, collect_novel=False):
        ch_dir = False

        print()

        try:

            os.chdir(self.dir_name)
            ch_dir = True
        except Exception as e:
            print('Error changing directory', e)
            ch_dir = False
            pass

        self.save_list(self.len_arr, 'len_', save_name)
        self.save_list(self.eval_arr, 'eval_', save_name)

        if self.offline:
            self.save_list(self.derived_arr, 'derived_', save_name)
        self.save_list(self.rt_arr, 'rt_', save_name)
        self.save_list(self.cummax_arr, 'cummax_', save_name)

        if self.offline and controller.kl_div:
            self.save_list(controller.kl_divergences, 'kl_', save_name)
            self.save_list(controller.kl_lens, 'kl_lens_', save_name)

        if is_boss:
            self.save_list(self.boss_schedule_list, 'schedule_', save_name)
            for k, v in controller.sampler_reward_dict.items():
                self.save_list(self.boss_rew_dict[k], k + '_rew_', save_name)
                self.save_list(self.boss_ucb_dict[k], k + '_ucb_', save_name)
                self.save_list(self.boss_ucb_dict[k], k + '_ucb_', save_name)

        if controller.save_traj:
            print('Trajectories:')
            self.save_list(self.trajectories, "full_trajectory_", save_name)

        if collect_novel:
            self.save_list(self.novel_sa, "novel_sa_occurances_", save_name)

        return ch_dir

    def moving_average(self, a, n=3):
        """
        Returns the moving average of series a with window n
        :param a: 1d list for moving average
        :param n: window
        :return: moving average of series a with window n
        """
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def _display_results(self, controller):
        """
        Creates master list of all results
        i.e creates a 2-d list containing all rewards lists for multiple runs
        :param controller:
        """
        agent_total_rewards = controller.callbacks[1].agent_total_rewards
        for agent, total_rewards in agent_total_rewards.items():
            self.rew_arr.append(total_rewards)
        self.len_arr.append(controller.callbacks[1].epi_lengths)



