"""
An example Controller.
"""
from typing import List, Tuple
import numpy as np
import copy
from config.moduleframe import AbstractModuleFrame
from config.config import ConfigItemDesc, ConfigDesc, ConfigListDesc
from agentsystem import AgentSystem, HierarchicalSystem
from sampler.PolledSampler import PolledSampler
from sampler.WeaklyPolledSampler import WeaklyPolledSampler
from config import Config, checks
from common.trajectory import TrajectoryCollection
from controller import MDPController
from algorithm.LSPI import LSPI
import pickle as pkl
from domain.conversion import FeatureConversions
from policy.function_approximator.basis_function import ExactBasis
import scipy.stats as stats
import time
import scipy as sp
import os
import sys
import warnings
from sampler.FlattenedPolledSampler import top_down_flatten, bottom_up_flatten
from sampler import BOSS
from domain.hierarchical_domain import HierarchicalActionDomain, action_hierarchy_from_config
import json
import math
from environment import Environment


class OfflineController(MDPController, AbstractModuleFrame):
    """
    Runs environment and learns in an offline manner, over episodes.
    Currently designed to only work with LSPI algorithm
    Offers several tuning and optimization algorithms for reducing the number of derived samples used during learning

    Expects a sampler

    @author: Eric Miller
    @contact: edm54@case.edu
    """
    def __init__(self, config: Config, iteration=0, current_date=''):
        sampling_hierarchies_map = {}
        self.eval_episode_max_length = config.eval_max_length
        self.is_etc = False
        if not hasattr(config, "sampler"):
            raise ValueError('Offline analyses require a sampler')

        if config.sampler.name == "FlattenedPolledSampler":
            self.derived_samples_hierarchy = copy.deepcopy(config.environment.action_hierarchy)

            iterations = 1
            if hasattr(config.sampler, 'iterations'):
                iterations = config.sampler.iterations
            keep_navigate = config.sampler.keep_navigate if hasattr(config.sampler, 'keep_navigate') else False

            # Modify hierarchy
            print('Flatten Hierarchy')
            if config.sampler.flattener == "BUF":
                h = bottom_up_flatten(config.environment.action_hierarchy['actions'],
                                      iterations=iterations,
                                      keep_navigate=keep_navigate)

            elif config.sampler.flattener == "TDF":
                h = top_down_flatten(config.environment.action_hierarchy['actions'],
                                     iterations=iterations,
                                     keep_navigate=keep_navigate)
            else:
                raise NotImplementedError('Only BUF and TDF supported')

            config.environment.action_hierarchy['actions'] = h
        elif config.sampler.name == "BOSS" or config.sampler.name == "ETC":
            self.is_etc = config.sampler.name == 'ETC'

            self.sampler_tag = "ETC" if self.is_etc else "BOSS"

            self.sampler_ind_dict = {}
            keep_navigate = config.sampler.keep_navigate if hasattr(config.sampler, 'keep_navigate') else False
            self.set_once = False
            for samplers in config.sampler.samplers_list:
                if "TDF" in samplers or "BUF" in samplers:
                    split_name = samplers.split('_')
                    iterations = int(split_name[-1])

                    if "TDF" in samplers:
                        h = top_down_flatten(config.environment.action_hierarchy['actions'],
                                             iterations=iterations,
                                             keep_navigate=keep_navigate)
                    else:
                        h = bottom_up_flatten(config.environment.action_hierarchy['actions'],
                                              iterations=iterations,
                                              keep_navigate=keep_navigate)

                    ah_copy = copy.deepcopy(config.environment.action_hierarchy)
                    ah_copy['actions'] = h
                    sampling_hierarchies_map[samplers] = ah_copy

            self.sampling_len_lst = []
            self.sampler_reward_dict = {}
            self.sampler_ucb_dict = {}
            self.sampler_schedule_arr = []
            self.sampler_ind_arr = []
            self.evaluation_environment: Environment = config.environment.module_class(config)
            self.evaluation_episode_step = 0
            self.eval_curr_trajectory: TrajectoryCollection = None
            self.sampler_policy_weights_dict = {}
            self.steps_per_sampler = config.sampler.steps_per_sampler
            # self.use_weights = config.sampler['use_weights']
            self.use_weights = False # Weights currently not supported

        else:
            self.use_weights = False
            self.steps_per_sample = None

        super(OfflineController, self).__init__(config)

        # Used to track metrics during learning
        self._init_sampler(config, sampling_hierarchies_map)
        self.rewards = []
        self.lens = []
        self.derived_samples = []
        self.rt = []
        self.kept = []
        self.sa_kl = []
        self.sprime_kl = []
        self.load = config.load
        self.save = config.save_samples
        self.name = config.samples_name

        self.current_sampler_steps = 0
        self.rotate = True
        self.converted_samples = []

        self.dir_name = self.get_dir_name(current_date, self.name)

        self.eval_samples = config.eval_samples
        self.episodes_to_run = config.episodes
        self.samples_target = config.samples_target
        self.sucessful_traj = []
        self.last_obs = []
        self.iteration = iteration # for naming only
        self.kl_only = config.kl_only
        self._init_basis()

        if hasattr(config.sampler, "collect_inhibited"):
            self.collect_inhibited = config.sampler.collect_inhibited
            self.collect_abstract = config.sampler.collect_abstract
        else:
            self.collect_abstract = False
            self.collect_inhibited = False

        if hasattr(config, 'display_distribution'):
            self.display_distribution = config.display_distribution
        else:
            self.display_distribution = False

        if config.kl_divergence:
            self.kl_div = True
            self.kl_divergences = []
            self.kl_samples = config.kl_divergence
            self.kl_lens = []
            self.entropies = []
            self.target_entropy = 0
        else:
            self.kl_div = False

        self.all_derived_samples = []
        self.sample_lens = []

    def _init_sampler(self, config, sampling_hierarchies_map=None):
        """
        Performs the initialization of the offline sampler
        Most of this is done for the BOSS algorithm
        For the flattened sampler, this will build the derived samples hierarchy
        """
        config_name = config.sampler.name
        print(config_name + ", has second hierarchy:", config.sampler.name == "FlattenedPolledSampler")

        if config_name == "BOSS" or config_name == "ETC":

            for k, h in sampling_hierarchies_map.items():
                flattened_action_hierarchy = action_hierarchy_from_config(h)
                flattened_hierarchy = self.env.load_hierarchy(flattened_action_hierarchy)
                flattened_hierarchical_action_domain = HierarchicalActionDomain(flattened_hierarchy.root,
                                                                                flattened_hierarchy)
                flattened_hierarchical_observation_domain = self.abstract_all_observation_domains(flattened_hierarchy)

                # Derived Hierarchy
                hierarchy_config_dict = config.environment.action_hierarchy
                action_hierarchy = action_hierarchy_from_config(hierarchy_config_dict)
                _ = self.env.load_hierarchy(action_hierarchy)
                _ = self.env.abstract_all_observation_domains()

                sampling_hierarchies_map[k] = (self.env.hierarchical_action_domain,
                                               self.env.hierarchical_observation_domain,
                                               flattened_hierarchical_action_domain,
                                               flattened_hierarchical_action_domain,
                                               flattened_hierarchical_observation_domain)

            self.sampler: AgentSystem = config.sampler.module_class(self.env.agent_class_map,
                                                                    self.env.agent_class_action_domains,
                                                                    self.env.agent_class_observation_domains,
                                                                    self.env.get_auxiliary_info(),
                                                                    config,
                                                                    sampling_hierarchies_map)
        else:

            if config.sampler.name == "FlattenedPolledSampler":
                self.env.create_second_hierarchy(self.derived_samples_hierarchy)

            self.sampler: AgentSystem = config.sampler.module_class(self.env.agent_class_map,
                                                                    self.env.agent_class_action_domains,
                                                                    self.env.agent_class_observation_domains,
                                                                    self.env.get_auxiliary_info(),
                                                                    config)
    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return [
                ConfigItemDesc('episode_max_length', checks.positive_integer, 'Max length of any episode'),
                ConfigDesc('environment', 'Environment config', default_configs=[]),
                ConfigDesc('agentsystem', 'Agentsystem config', default_configs=[]),
                ConfigDesc('policy', 'Policy config', default_configs=[]),
                ConfigDesc('algorithm', 'Algorithm config', default_configs=[]),
                ConfigDesc('sampler', 'AgentSystem for collecting samples', default_configs=[]),
                ConfigListDesc('callbacks', 'List of callback configs', default_configs=[]),
                ConfigItemDesc('num_trajectories', checks.nonnegative_integer,
                               'How many past trajectories to keep in memory.'
                               'Trajectories may take up a lot of RAM if kept in memory.'
                               '0 to keep all trajectories (use only if you are certain).'
                               '\nDefault: 5.', default=5, optional=True),
                ConfigItemDesc('seed', checks.nonnegative_integer, 'Seed for random variables.\n Default: 2',
                               default=2, optional=True),
                ConfigItemDesc('samples_target', checks.positive_integer, 'Number of samples to gather',
                               default=-1, optional=True),
                ConfigItemDesc('episodes', checks.positive_integer, 'Number of episodes to run',
                               default=100, optional=True),
                ConfigItemDesc('eval_samples', checks.positive_int_list, 'List of episodes to evaluate',
                               default=-1, optional=True),
                ConfigItemDesc('eval_num_samples', checks.positive_int_list, 'Number of samples to evaluate',
                               default=-1, optional=True),
                ConfigItemDesc('save_samples', checks.boolean, 'Saves samples for future testing',
                               default=False, optional=True),
                ConfigItemDesc('samples_name', checks.string, 'Name for saving samples for future testing',
                                default='', optional=True),
                ConfigItemDesc('load', checks.boolean, 'Load previously saved examples ',
                               default=False, optional=True),
                ConfigItemDesc('keep_fraction', checks.nonnegative_float,
                               'The fraction of derived samples to keep, or the amount of samples the tuner starts at',
                               default=0.5, optional=True),
                ConfigItemDesc('use_tuner', checks.boolean, 'Use algorithm to select optimal fraction to keep',
                               default=True, optional=True),
                ConfigItemDesc('save_traj', checks.boolean, 'Accumulate all primitive actions',
                               default=False, optional=True),
                ConfigItemDesc('novel_states_count', checks.boolean,
                               'Tracks the number of novel states, action visited',
                               default=False, optional=True),
                ConfigItemDesc('display_distribution', checks.boolean, 'Display SA Distribution for KL divergence',
                               default=False, optional=True),
                ConfigItemDesc('kl_divergence', checks.positive_int_list,
                               'Compare samples distribution to target distribution at specified # of samples',
                               default=[], optional=True),
                ConfigItemDesc('kl_only', checks.boolean,
                               'Only get KL, do not learn',
                               default=False, optional=True),
            ConfigItemDesc('eval_max_length', checks.positive_integer, 'Max length of any eval episode',
                           default=50, optional=True)

                ]

    def run(self):
        """
        This method runs the analysis for all but the BOSS algorithm.
        First all of the samples are collected, then the policy is learned
        """

        if self.config.sampler.name == "BOSS" or self.config.sampler.name == 'ETC':
            self.run_BOSS()
        else:
            self.before_run()
            samples = []
            num_samples = 0
            episode_num = 1
            kl_samples = self.kl_samples[:] if self.kl_div else []
            sampling_times = []
            sampling_lens = []
            episode_max_length = copy.copy(self.episode_max_length)
            derived_len = []
            d_sum = 0

            # Can either run until samples target or episodes target, set samples to -1 to hit episodes target
            if not self.load:
                start_time = time.perf_counter()
                use_sample_num = True if self.config.eval_num_samples != -1 else False
                sample_lens = self.config.eval_num_samples[:] if use_sample_num else self.config.eval_samples[:]
                orig_samples_len = copy.copy(sample_lens)

                print('Beginning Sampling!')
                print('==================================================')
                while self.continue_running(num_samples, sample_lens, episode_num, orig_samples_len):

                    self.initialize_episode(num_samples, episode_num)

                    episode_trajectory, last_obs, done, kl_samples = self.run_sample(kl_samples, num_samples)
                    episode_trajectory.cull()
                    samples.append(episode_trajectory)

                    self._append_trajectories(episode_trajectory, last_obs, done)

                    # Get derived samples for episode
                    if self.collect_inhibited or self.collect_abstract:
                        num_derived = self._update_derived(last_obs, done, not use_sample_num)
                        d_sum += num_derived

                    self._save_traj(episode_trajectory)
                    self.after_episode()

                    num_samples += episode_trajectory.time
                    episode_num += 1

                    if kl_samples and self.kl_div and num_samples >= kl_samples[0]:
                        kl_samples.pop(0)
                        self._append_kl_divergence(num_samples)

                    if orig_samples_len != [] and \
                            (num_samples >= sample_lens[0] or self.samples_target <= num_samples):
                        sampling_times.append(time.perf_counter() - start_time)
                        sampling_lens.append(num_samples)
                        sample_lens.pop(0)
                        derived_len.append(d_sum)

                self.sampler.reset()
                self.episode_max_length = episode_max_length

                if hasattr(self.sampler, "save_target") and self.sampler.save_target:
                    current_dist = self.sampler.get_distribution()
                    self.sampler.save_dist(current_dist)
                    print('Saved target')

                print('Begin to learn on {} samples...'.format(len(samples)))
                converted_samples = self.convert_trajectory(samples)

                if self.config.sampler.name != "BOSS":
                    converted_samples = self.map_all(converted_samples)

                self._get_derived_samples_per_sample()
                self._update_novel_states_count(converted_samples)

            else:
                samples = []
                converted_samples = []

            self.episode_max_length = min(episode_max_length, self.eval_episode_max_length)
            self._delete_after_run()

            # All of the samples are collected so its time to learn!
            if self.config.use_tuner:
                self._fraction_tuner(converted_samples, samples, starting_fraction=self.config.keep_fraction)
            elif not self.kl_only and (self.config.eval_samples != -1 or \
                                        (hasattr(self.config, 'eval_num_samples') and \
                                         self.config.eval_num_samples != -1)):
                self.evaluate(converted_samples, samples, add_times=sampling_times)
            self.after_run()
            return self.finalize()

    def run_BOSS(self):
        """
        Ran an analysis for the BOSS algorithm
        Requires its own algorithm since evaluation is done during sample collection
        :return:
        """
        self.before_run()
        samples = []
        num_samples = 0
        episode_num = 1
        kl_samples = self.kl_samples[:] if self.kl_div else []
        sampling_times = []
        episode_max_length = copy.copy(self.episode_max_length)
        d_sum = 0
        sa_ind_samples_per_sampler = {}
        sa_ind_derived_samples_per_sampler = {}

        # Can either run until samples target or episodes target, set samples to -1 to hit episodes target
        if not self.load:
            start_time = time.perf_counter()
            use_sample_num = True if self.config.eval_num_samples != -1 else False
            sample_lens = self.config.eval_num_samples[:] if use_sample_num else self.config.eval_samples[:]
            orig_samples_len = copy.copy(sample_lens)

            sa_ind_samples = []
            sa_ind_derived_samples = []

            weights_dict = {}
            derived_weights_dict = {}

            sampler_probability_map = {}
            time_to_subtract = 0
            print('Beginning Sampling!')

            while self.continue_running(num_samples, sample_lens, episode_num, orig_samples_len):

                self.sampler_ind_dict = {}
                self.initialize_episode(num_samples, episode_num)

                (episode_trajectory,
                 last_obs, done, num_samples,
                 not_added_index, time_to_subtract) = self.run_boss_sample(num_samples,
                                                                           sample_lens,
                                                                           sa_ind_samples,
                                                                           sa_ind_derived_samples,
                                                                           sa_ind_samples_per_sampler,
                                                                           sampler_probability_map,
                                                                           episode_num, weights_dict,
                                                                           derived_weights_dict,
                                                                           sa_ind_derived_samples_per_sampler,
                                                                           start_time, sampling_times, d_sum,
                                                                           kl_samples, time_to_subtract)

                episode_trajectory.cull()
                samples.append(episode_trajectory)

                self._append_trajectories(episode_trajectory, last_obs, done)
                start_time2 = time.perf_counter()
                converted_traj = self.convert_trajectory([episode_trajectory])[0].tolist()[not_added_index:]

                if self.use_weights:
                    sampler_prob_map = self.make_probability_map([converted_traj])

                mapped_s = self.map_all([converted_traj])[0]
                time_to_subtract += (time.perf_counter() - start_time2)

                # Finish converting all collected samples in the last episode
                if mapped_s != []:

                    start_time2 = time.perf_counter()

                    # Converts samples to basis function samples
                    evaluated_samples = self.eval_all_samples(mapped_s)
                    sa_ind_samples.extend(evaluated_samples)
                    self._check_sampler_distribution_macro(mapped_s)

                    sa_ind_samples_per_sampler = self.extend_sa_samples(sa_ind_samples_per_sampler,
                                                                        not_added_index,
                                                                        mapped_s,
                                                                        evaluated_samples)

                    if self.use_weights:
                        single_weights_dict, weights_dict, derived_weights_dict = self.calculate_weights(sampler_prob_map,
                                                                                                         sampler_probability_map,
                                                                                                         episode_num,
                                                                                                         weights_dict,
                                                                                                         derived_weights_dict)

                    # Get derived samples for episode
                    if self.collect_inhibited or self.collect_abstract:
                        num_derived = self._update_derived(last_obs, done, not use_sample_num)
                        d_sum += num_derived

                    single_episode_derived = self.sampler.current_ep_derived

                    if self.use_weights:
                        sa_ind_derived_samples, \
                        derived_weights_dict = self.extend_sa_derived_samples(single_episode_derived,
                                                                              sa_ind_derived_samples_per_sampler,
                                                                              sa_ind_derived_samples,
                                                                              not_added_index,
                                                                              single_weights_dict,
                                                                              derived_weights_dict)
                    else:
                        sa_ind_derived_samples = self.extend_sa_derived_samples(single_episode_derived,
                                                                                   sa_ind_derived_samples_per_sampler,
                                                                                   sa_ind_derived_samples,
                                                                                   not_added_index)

                    time_to_subtract += (time.perf_counter() - start_time2)

                self._save_traj(episode_trajectory)
                self.after_episode()

                episode_num += 1

                self.sampler.reset_current_ep_derived()
                self.sampler.reset_sampler_index_list()

                if sample_lens and num_samples >= sample_lens[0]:
                    self.derived_samples.append(num_derived)
                    # if self.use_weights:
                    eval_time = self.eval_boss(sa_ind_samples,
                                                sa_ind_derived_samples,
                                                sa_ind_samples_per_sampler,
                                                weights_dict=weights_dict,
                                                derived_weights_dict=derived_weights_dict)

                    time_to_subtract += eval_time
                    sampling_times.append(time.perf_counter() - start_time - time_to_subtract)
                    self.rt.append(time.perf_counter() - start_time - time_to_subtract)
                    sample_lens.pop(0)

                if kl_samples and self.kl_div and num_samples >= kl_samples[0]:
                    kl_samples.pop(0)
                    self._append_kl_divergence(num_samples)

            self.sampler.reset()
            self.episode_max_length = episode_max_length

            if hasattr(self.sampler, "save_target") and self.sampler.save_target:
                current_dist = self.sampler.get_distribution()
                self.sampler.save_dist(current_dist)
        else:
            raise NotImplementedError

        self.episode_max_length = min(episode_max_length, 500)
        self.after_run()
        return self.finalize()

    def evaluate(self, converted_samples, samples, add_times=None):
        """
        * use_sample_num means that a specific number of samples is being evaluated,
        as opposed to a specific number of episodes

        Evaluates the learning algorithm on a series of predefined sample counts
        This is for all but the BOSS algorithm

        :param converted_samples: samples that have been converted to a list of [s,a,r,s')
        :param samples: samples that have not been converted to this form yet.
        """
        use_sample_num = True if self.config.eval_num_samples != -1 else False
        sample_lens = self.config.eval_num_samples if use_sample_num else self.config.eval_samples
        all_samples = []

        if not self.load:
            if use_sample_num:
                all_samples = np.asarray(self._create_single_list(converted_samples), dtype=object)

            sa_ind_samples, sa_ind_derived_samples, time_to_convert = self._sa_ind_all_samples(all_samples)

            # To save space we can deallocate these
            if use_sample_num:
                del converted_samples
            del all_samples
            del self.derived_samples_per_sample

            if self.save:
                self.save_samples(sa_ind_samples, sa_ind_derived_samples)

        else:
            time_to_convert = 0
            sa_ind_samples, sa_ind_derived_samples = self.load_samples()

        for i in sample_lens:

            if use_sample_num:
                num_samps = min(i, len(sa_ind_samples))
                current_samples = sa_ind_samples[:num_samps]

                single_list = True
                if self.collect_inhibited or self.collect_abstract:

                    # Get all derived samples for the sample range
                    dev_samples = sa_ind_derived_samples[:num_samps]
                    all_dev_samples = np.asarray(self._create_single_list(dev_samples), dtype=object)

                    self.derived_samples.append(len(all_dev_samples))
                    if 1.0 >self.config.keep_fraction >0:
                        all_dev_samples = self._remove_redundant_samples([all_dev_samples], fraction=self.config.keep_fraction)[0]

                    try:
                        current_samples = np.concatenate((sa_ind_samples[:num_samps], all_dev_samples))
                    except:
                        raise ValueError('Unable to append derived samples')

                self.lens.append(num_samps)

            # A specific number of episodes is used
            else:
                self._get_num_samples(samples, i)
                # Get samples for testing range
                current_samples = self._get_samples(converted_samples[:i], i, fraction=self.config.keep_fraction)
                single_list = False

            start = time.time()

            # Get Policy
            self.asys.algorithm.learn(current_samples, self.asys.policy_groups, single_list=single_list)
            end = time.time()

            current_reward = self.eval_policy(25)
            if add_times:
                samples_ratio = min(i/self.samples_target, 1)
                time_to_app = end - start + samples_ratio * add_times[-1] + samples_ratio * time_to_convert

            else:
                time_to_app = end - start
            self._append_all(current_reward, self.rewards, time_to_app, self.rt)
            self._display_arrays()

    def eval_all_samples(self, samples:list):
        """
        1) Converts a list of samples to their converted basis function
        2) Creates a SA,R,SA' sample with S, S' representing the state-action pair from the basis
        converted to the basis function
        3) for s', we use all possible s' since this is what is used in learning for Q(s', a')
        :param samples: list of samples collected to be converted
        :return: converted samples
        """
        samples = np.asarray(samples, dtype=object)
        for ind, sample in enumerate(samples):
            _, sa_ind = self.basis.evaluate2(sample[0], sample[1])
            samples[ind][0] = sa_ind
            if sample[3] is not None:
                all_ind = self.basis.get_state_action_index_batch(sample[3], range(self.basis.num_actions))
                samples[ind][3] = all_ind.astype('int32')
            else:
                samples[ind][3] = []
        samples = np.delete(samples, 1, 1)
        return samples

    def eval_all_derived_samples(self, samples, weights=None):
        """
           1) Converts a list of derived samples to their converted basis function
           2) Creates a SA,R,SA' sample with S, S' representing the state-action pair from the basis
           converted to the basis function
           3) for s', we use all possible s' since this is what is used in learning for Q(s', a')
           4) sets weights if needed
           :param samples: list of samples collected to be converted
           :return: converted samples of length num regular samples where each index is a list containing
           all of the derived samples corresponding to that singluar collected sample
        """
        derived_weights = {}
        for j, timestep_list in enumerate(samples):
            samples[j] = np.asarray(samples[j], dtype=object)
            for ind, sample in enumerate(timestep_list):
                _, sa_ind = self.basis.evaluate2(sample[0], sample[1])
                samples[j][ind][0] = int(sa_ind)

                if sample[3] is not None:
                    all_ind = self.basis.get_state_action_index_batch(sample[3], range(self.basis.num_actions))
                    samples[j][ind][3] = all_ind.astype('int32')
                else:
                    samples[j][ind][3] = []

                if weights:
                    for s in self.sampler.sampler_object_dict.keys():
                        if s not in derived_weights:
                            derived_weights[s] = []

                        derived_weights[s].append(weights[s][j])

            if len(samples[j]) > 0:
                samples[j] = np.delete(samples[j], 1, 1)

        if weights:
            return np.array(samples, dtype=object), derived_weights

        else:
            return np.array(samples, dtype=object)

    def eval_policy_boss(self,  sa_ind_samples, sa_ind_derived_samples, sa_ind_samples_per_sampler,
                        sa_ind_derived_samples_per_sampler=None, sampler=None, num_episodes=25, use_eval_environment=False, return_len=False):
        """
        Learns a policy based on the samples and evaluates it.
        :return:
        """
        if hasattr(self.asys, 'hierarchy'):
            raise TypeError('Hierarchical agent systems only supported for sampling')
        elif not isinstance(self.asys.algorithm, LSPI):
            raise TypeError('LSPI is the only offline learning algorithm currently supported')

        rewards, len_traj = self.run_eval_samples_boss( sa_ind_samples, sa_ind_derived_samples,
                                                   sa_ind_samples_per_sampler,
                                                   sa_ind_derived_samples_per_sampler,
                                                   sampler,
                                                   num_episodes=num_episodes,
                                                   use_eval_environment=use_eval_environment)
        self.after_run()

        if return_len:
            return rewards / num_episodes, len_traj
        else:
            return rewards/num_episodes

    def eval_policy(self, num_episodes=25):
        """
        Learns a policy based on the samples and evaluates it.
        :return:
        """
        if hasattr(self.asys, 'hierarchy'):
            raise TypeError('Hierarchical agent systems only supported for sampling')
        elif not isinstance(self.asys.algorithm, LSPI):
            raise TypeError('LSPI is the only offline learning algorithm currently supported')

        rewards, len_traj = self.run_eval_samples()
        self.after_run()

        if return_len:
            return rewards / num_episodes, len_traj
        else:
            return rewards/num_episodes

    def run_boss_sample(self, num_samples, sample_lens, sa_ind_samples, sa_ind_derived_samples,
                         sa_ind_samples_per_sampler, sampler_probability_map,
                         episode_num, weights_dict, derived_weights_dict, sa_ind_derived_samples_per_sampler,
                         start_time, sampling_times, d_sum, kl_samples, time_to_subtract):
        """

        :param num_samples: number of samples already collected
        :param sample_lens: list holding the indexes to evaluate at
        :param sa_ind_samples: list of collected samples that have been converted to basis function s, s'
        :param sa_ind_derived_samples: list of collected derived samples that have been converted to basis function s, s'
        :param sa_ind_samples_per_sampler: a dictionary mapping each sampler to its list of collected samples
        :param sampler_probability_map: maps the probability of each sample (only for weights)
        :param episode_num: current episode number
        :param weights_dict: dict of weights
        :param derived_weights_dict: dict of weights of derived samples
        :param sa_ind_derived_samples_per_sampler: converted samples in dictionary for each sampler
        :param start_time: time the analysis started for runtime analysis
        :param sampling_times: list of runtimes for each analysis threshold
        :param d_sum: number of derived samples
        :param kl_samples: would be used to hold kl
        :param time_to_subtract: allows to adjust for correct runtime
        :return:
        """

        self.episode_step = 0
        self.curr_trajectory: TrajectoryCollection = TrajectoryCollection(self.episode_max_length,
                                                                          self.env.agent_class_map,
                                                                          self.env.agent_class_action_domains,
                                                                          self.env.agent_class_observation_domains)
        self.env.reset(self.flags.visualize)
        self.sampler.reset()
        self.on_episode_start()
        sampler_start = 0
        use_sample_num = True if self.config.eval_num_samples != -1 else False
        not_added_index = 0

        samples_collected_with_sampler = 0
        previous_stoppage = 0
        append=True

        for k, v in self.sampler_ind_dict.items():
            self.sampler_ind_dict[k] = []

        while self.episode_active():

            self.sample_step(append=append)
            append = True
            samples_collected_with_sampler += 1
            num_samples += 1

            self.current_sampler_steps += 1
            self.sampler_ind_arr.append(self.sampler.current_sampler_name)

            # Check for evaluation
            if num_samples >= sample_lens[0]:

                self.append_ind_list(sampler_start)
                sampler_start = self.episode_step

                episode_trajectory = copy.deepcopy(self.curr_trajectory)
                episode_trajectory.cull()
                observation_request = self.sampler.observe_request()
                last_obs = self.env.observe(observation_request)

                start_time2 = time.perf_counter()
                converted_traj = self.convert_trajectory([episode_trajectory], last_obs=last_obs)[0].tolist()[previous_stoppage:]

                if self.use_weights:
                    sampler_prob_map = self.make_probability_map([converted_traj])

                mapped_s = self.map_all([converted_traj])[0]
                evaluated_samples = self.eval_all_samples(mapped_s)

                self._check_sampler_distribution_macro(mapped_s)
                sa_ind_samples.extend(evaluated_samples)

                sa_ind_samples_per_sampler = self.extend_sa_samples(sa_ind_samples_per_sampler,
                                                                       not_added_index,
                                                                       mapped_s,
                                                                       evaluated_samples)
                single_weights_dict = None
                if self.use_weights:
                    single_weights_dict, weights_dict, derived_weights_dict = self.calculate_weights(sampler_prob_map,
                                                                                                     sampler_probability_map,
                                                                                                     episode_num,
                                                                                                     weights_dict,
                                                                                                     derived_weights_dict)

                observation_request = self.sampler.observe_request()
                last_obs = self.env.observe(observation_request)
                done = False

                if self.collect_inhibited or self.collect_abstract:
                    num_derived = self._update_derived(last_obs, done, not use_sample_num)
                    append = False

                single_episode_derived = self.sampler.current_ep_derived[previous_stoppage:]

                if self.use_weights:
                    sa_ind_derived_samples, \
                    derived_weights_dict = self.extend_sa_derived_samples(single_episode_derived,
                                                                          sa_ind_derived_samples_per_sampler,
                                                                          sa_ind_derived_samples,
                                                                          not_added_index,
                                                                          single_weights_dict,
                                                                          derived_weights_dict)
                else:
                    sa_ind_derived_samples = self.extend_sa_derived_samples(single_episode_derived,
                                                                            sa_ind_derived_samples_per_sampler,
                                                                            sa_ind_derived_samples,
                                                                            not_added_index)

                time_to_subtract += (time.perf_counter() - start_time2)

                if self.is_etc and self.set_once:
                    eval_time = self.eval_etc(sa_ind_samples,
                                               sa_ind_derived_samples)

                else:
                    eval_time =self.eval_boss(sa_ind_samples,
                                                sa_ind_derived_samples,
                                                sa_ind_samples_per_sampler,
                                                sa_ind_derived_samples_per_sampler,
                                                weights_dict,
                                                derived_weights_dict)

                time_to_subtract += eval_time
                collection_time = time.perf_counter() - start_time - time_to_subtract

                sampling_times.append(collection_time)
                self.rt.append(collection_time)
                sample_lens.pop(0)

                if self.collect_inhibited or self.collect_abstract:
                    d_sum += num_derived

                if kl_samples and self.kl_div and num_samples >= kl_samples[0]:
                    kl_samples.pop(0)
                    self._append_kl_divergence(num_samples)

                previous_stoppage = copy.deepcopy(self.episode_step)
                not_added_index = len(episode_trajectory)

            if self.current_sampler_steps >= self.steps_per_sampler and self.rotate:
                self.append_ind_list(sampler_start)
                self.sampler.set_next_sampler()
                self.current_sampler_steps = 0
                sampler_start = copy.deepcopy(self.episode_step)

        observation_request = self.sampler.observe_request()
        last_obs = self.env.observe(observation_request)

        if self.flags.visualize:
            self.env.visualize()

        self.sampler.end_episode(self.flags.learn)
        self.append_ind_list(sampler_start)

        return self.curr_trajectory,last_obs, self.env.done, num_samples, not_added_index, time_to_subtract

    def calculate_weights(self, sampler_prob_map, sampler_probability_map,
                          episode_num, weights_dict, derived_weights_dict):

        """
        Calculates weights of samples using probability of each sample given the sampler
        :return:
        """

        for k, v in sampler_prob_map.items():
            if k not in sampler_probability_map:
                sampler_probability_map[k] = []
            sampler_probability_map[k].extend(v)

        single_weights_dict = {}

        for ind, source in enumerate(self.sampler.sampler_index_list):
            for sampler_name, obj in self.sampler.sampler_object_dict.items():

                if ind == 0:
                    single_weights_dict[sampler_name] = []
                if ind == 0 and episode_num == 1:
                    weights_dict[sampler_name] = []
                    derived_weights_dict[sampler_name] = []

                numerator = sampler_probability_map[sampler_name][ind]
                weights_dict[sampler_name].append(numerator)
                single_weights_dict[sampler_name].append(numerator)

        return single_weights_dict, weights_dict, derived_weights_dict

    def eval_boss(self, sa_ind_samples, sa_ind_derived_samples, sa_ind_samples_per_sampler,
                   sa_ind_derived_samples_per_sampler=None, weights_dict=None, derived_weights_dict=None):
        """

        ::param sa_ind_samples: list of collected samples that have been converted to basis function s, s'
        :param sa_ind_derived_samples: list of collected derived samples that have been converted to basis function s, s'
        :param sa_ind_samples_per_sampler: a dictionary mapping each sampler to its list of collected samples
        :param sa_ind_derived_samples_per_sampler: converted samples in dictionary for each sampler
        :param weights_dict:
        :param derived_weights_dict:
        :return:
        """
        assert not (self.set_once and self.is_etc)

        episode_max_length = copy.copy(self.episode_max_length)
        self.rotate = False

        reward_dict = {}
        ucb_dict = {}

        num_samples = len(sa_ind_samples)

        max_sampler_ucb = -1e99
        max_sampler_obj = None
        max_sampler_name = None
        current_samples = np.concatenate((sa_ind_samples, sa_ind_derived_samples))
        time_to_subtract = 0

        if self.sampler_tag in self.sampler_policy_weights_dict:
            self.asys.policy_groups[0].policy.function_approximator.set_weights(
                self.sampler_policy_weights_dict[self.sampler_tag])

        # # DO BOSS policy first
        self.asys.algorithm.learn(current_samples,
                                  self.asys.policy_groups,
                                  single_list=True,
                                  sampler=self.sampler_tag)

        reward, traj_len = self.eval_policy_boss( sa_ind_samples, sa_ind_derived_samples, sa_ind_samples_per_sampler,
                                             sa_ind_derived_samples_per_sampler=sa_ind_derived_samples_per_sampler,
                                             sampler="BOSS", num_episodes=3,use_eval_environment=True, return_len=True)

        self.sampling_len_lst.append(traj_len)
        self.sampler_policy_weights_dict[self.sampler_tag] = self.asys.policy_groups[0].\
                                                                policy.\
                                                                function_approximator.\
                                                                get_weights()

        reward_dict[self.sampler_tag] = reward

        if self.sampler_tag not in self.sampler_reward_dict:
            self.sampler_reward_dict[self.sampler_tag] = []

        self.sampler_reward_dict[self.sampler_tag].append(reward)

        for sampler_name, obj in self.sampler.sampler_object_dict.items():
            if sampler_name in sa_ind_samples_per_sampler:
                if sampler_name not in self.sampler_reward_dict or \
                      sampler_name == self.sampler.current_sampler_name:

                    if sa_ind_derived_samples_per_sampler and sa_ind_derived_samples_per_sampler[sampler_name]:
                        current_samples = np.concatenate((sa_ind_samples_per_sampler[sampler_name],
                                                          sa_ind_derived_samples_per_sampler[sampler_name]))
                    else:
                        current_samples = sa_ind_samples_per_sampler[sampler_name]
                    weight_list = None
                    if self.use_weights:
                        weight_list = self._write_weights(sampler_name, weights_dict, derived_weights_dict)

                    if sampler_name in self.sampler_policy_weights_dict:
                        self.asys.policy_groups[0].policy.function_approximator.set_weights(
                            self.sampler_policy_weights_dict[sampler_name])

                    self.asys.algorithm.learn(current_samples,
                                              self.asys.policy_groups,
                                              single_list=True,
                                              weights=weight_list,
                                              sampler=sampler_name)

                    self.sampler_policy_weights_dict[sampler_name] = self.asys.policy_groups[0].policy.function_approximator.get_weights()

                    reward, time_to_subtract = self.eval_and_time_boss(sa_ind_samples,
                                                                       sa_ind_derived_samples,
                                                                       sa_ind_samples_per_sampler,
                                                                       sa_ind_derived_samples_per_sampler=sa_ind_derived_samples_per_sampler,
                                                                       sampler=sampler_name,
                                                                       num_episodes=3)
                    reward_dict[sampler_name] = reward
                else:
                    reward = self.sampler_reward_dict[sampler_name][-1]
                    reward_dict[sampler_name] = reward

                ucb_multiplier = self.sampler.ucb_coef
                upper_bound = ucb_multiplier * math.sqrt(((self.env.reward_range**2)
                                                          * math.log(0.005, 2))/
                                                         (-2 * (max(self.sampler.sampler_occurance_dict[sampler_name], .1))))
                ucb_dict[sampler_name] = reward + upper_bound
                if ucb_dict[sampler_name] > max_sampler_ucb:
                    max_sampler_obj = obj
                    max_sampler_name = sampler_name
                    max_sampler_ucb = ucb_dict[sampler_name]

                if sampler_name not in self.sampler_reward_dict.keys():
                    self.sampler_reward_dict[sampler_name] = []
                    self.sampler_ucb_dict[sampler_name] = []
                self.sampler_reward_dict[sampler_name].append(reward)
                self.sampler_ucb_dict[sampler_name].append(ucb_dict[sampler_name])

        self._set_boss_sampler(max_sampler_obj, max_sampler_name)
        self.set_once = True

        self.lens.append(num_samples)
        self.rewards.append(reward_dict[self.sampler_tag])
        self.episode_max_length = episode_max_length
        return time_to_subtract

    def eval_etc(self, sa_ind_samples, sa_ind_derived_samples):
        """

        ::param sa_ind_samples: list of collected samples that have been converted to basis function s, s'
        :param sa_ind_derived_samples: list of collected derived samples that have been converted to basis function s, s'
        :param sa_ind_samples_per_sampler: a dictionary mapping each sampler to its list of collected samples
        :param sa_ind_derived_samples_per_sampler: converted samples in dictionary for each sampler
        :param weights_dict:
        :param derived_weights_dict:
        :return:
        """
        episode_max_length = copy.copy(self.episode_max_length)
        self.rotate = False

        reward_dict = {}
        num_samples = len(sa_ind_samples)
        current_samples = np.concatenate((sa_ind_samples, sa_ind_derived_samples))
        time_to_subtract = 0

        if self.sampler_tag in self.sampler_policy_weights_dict:
            self.asys.policy_groups[0].policy.function_approximator.set_weights(
                self.sampler_policy_weights_dict[self.sampler_tag])

        # # DO BOSS policy first
        self.asys.algorithm.learn(current_samples,
                                  self.asys.policy_groups,
                                  single_list=True,
                                  sampler=self.sampler_tag)

        reward = self.eval_policy(num_episodes=25, use_eval_environment=True)
        self.sampler_policy_weights_dict[self.sampler_tag] = self.asys.policy_groups[0].\
                                                                policy.\
                                                                function_approximator.\
                                                                get_weights()

        reward_dict[self.sampler_tag] = reward

        if self.sampler_tag not in self.sampler_reward_dict:
            self.sampler_reward_dict[self.sampler_tag] = []

        self.sampler_reward_dict[self.sampler_tag].append(reward)
        self.etc_extend_samplers()

        self.lens.append(num_samples)
        self.rewards.append(reward_dict[self.sampler_tag])
        self.episode_max_length = episode_max_length
        return time_to_subtract


    def etc_extend_samplers(self):
        self.sampler_schedule_arr.append(self.sampler.current_sampler_name)
        print('Selected:', self.sampler.current_sampler_name)
        self.sampler.sampler_occurance_dict[self.sampler.current_sampler_name] += 1
        for  k, v in self.sampler_reward_dict.items():
            if k != self.sampler_tag:
                self.sampler_reward_dict[k].append(v[-1])
                self.sampler_ucb_dict[k].append(self.sampler_ucb_dict[k][-1])

    def convert_trajectory(self, samples, agent_id=0, last_obs=None):
        """
        Convert samples to a list in form [[s1, a1, r1, s1'], ...]
        Used so actual samples match the format of the derived samples

        :param samples: samples to convert
        :param agent_id: agent id of trajectory
        :return: list of samples in form [[s1, a1, r1, s1'], ...]
        """
        samples_list = [None] * len(samples)
        for ind, episode in enumerate(samples):
            traj = episode.get_agent_trajectory(agent_id)
            episode_sucess = traj.done
            if not last_obs:
                next_obs = self.last_obs[ind]
            else:
                next_obs = last_obs
            max_t = episode.time
            episode_list = [None] * max_t

            episode = episode.get_agent_trajectory(agent_id)
            for index in range(max_t):

                if index != max_t - 1:
                    s_prime = episode.observations[index+1]
                elif episode_sucess:
                    s_prime = None
                else:
                    s_prime = next_obs[0]

                sample = np.asarray([episode.observations[index],
                                     episode.actions[index],
                                     episode.rewards[index],
                                     s_prime],
                                    dtype=object)
                episode_list[index] = sample
            samples_list[ind] = episode_list

        return np.asarray(samples_list, dtype=object)

    def run_sample(self, kl_samples, num_samples):
        """
        Run a sample to collect samples
        """

        self.episode_step = 0
        self.curr_trajectory: TrajectoryCollection = TrajectoryCollection(self.episode_max_length,
                                                                          self.env.agent_class_map,
                                                                          self.env.agent_class_action_domains,
                                                                          self.env.agent_class_observation_domains)
        self.env.reset(self.flags.visualize)
        self.sampler.reset()
        self.on_episode_start()

        while self.episode_active():
            self.sample_step()
            if kl_samples and self.kl_div and num_samples + self.episode_step >= kl_samples[0]:
                kl_samples.pop(0)
                self._append_kl_divergence(num_samples + self.episode_step)

        observation_request = self.sampler.observe_request()
        last_obs = self.env.observe(observation_request)

        if self.flags.visualize:
            self.env.visualize()

        self.sampler.end_episode(self.flags.learn)
        return self.curr_trajectory, last_obs, self.env.done, kl_samples

    def sample_step(self, append=True):
        """
        Advance sampler one step
        :param append:
        :return:
        """
        transfer_msg = self.env.pop_last_domain_transfer()
        if transfer_msg is not None:
            self.sampler.transfer_domain(transfer_msg)

        observation_request = self.sampler.observe_request()

        self.before_observe()
        a_obs = self.env.observe(observation_request)
        a_obs = self.on_observe(a_obs)

        if self.flags.visualize:
            self.env.visualize()

        if isinstance(self.sampler, BOSS.BOSS):
            a_act = self.sampler.get_actions(a_obs, self.flags.exploit, append=append)
        else:
            a_act = self.sampler.get_actions(a_obs, self.flags.exploit)
        a_obs, a_act = self.on_action(a_obs, a_act)

        a_rew = self.env.update(a_act)
        update_signal = self.on_update(a_obs, a_act, a_rew)
        a_obs, a_act, a_rew = update_signal

        # Reshape data via asys for asys
        self.curr_trajectory.append(self.sampler.agent_ids, a_obs, a_act, a_rew, self.env.done)
        self.sampler.append_pg_signals(a_obs, a_act, a_rew, self.env.done)

        if isinstance(self.sampler, HierarchicalSystem.HierarchicalSystem) or \
                isinstance(self.sampler, PolledSampler) or isinstance(self.sampler, WeaklyPolledSampler):
            a_obs2 = self.env.observe(observation_request)
            self.sampler.check_all_agent_termination(a_obs2)

        self.after_update()
        self.episode_step += 1

    def eval_environment_step(self):
        """
        Advance sampler one step in the eval environment
        :return:
        """
        transfer_msg = self.evaluation_environment.pop_last_domain_transfer()
        if transfer_msg is not None:
            self.sampler.transfer_domain(transfer_msg)

        observation_request = self.sampler.observe_request()

        self.before_observe()
        a_obs = self.evaluation_environment.observe(observation_request)
        a_obs = self.on_observe(a_obs)

        if self.flags.visualize:
            self.evaluation_environment.visualize()

        a_act = self.asys.get_actions(a_obs, self.flags.exploit)
        a_obs, a_act = self.on_action(a_obs, a_act)

        a_rew = self.evaluation_environment.update(a_act)
        update_signal = self.on_update(a_obs, a_act, a_rew)
        a_obs, a_act, a_rew = update_signal

        # Reshape data via asys for asys
        self.eval_curr_trajectory.append(self.sampler.agent_ids, a_obs, a_act, a_rew, self.evaluation_environment.done)

        if isinstance(self.asys, HierarchicalSystem.HierarchicalSystem) or \
                isinstance(self.asys, PolledSampler):
            a_obs2 = self.evaluation_environment.observe(observation_request)
            self.asys.check_all_agent_termination(a_obs2)

        self.after_update()
        self.evaluation_episode_step += 1

    def eval_episode_active(self):
        """
        :return: Whether or not the episode is active (that is, can proceed to the next step).
        """
        return not self.evaluation_environment.done and (self.evaluation_episode_step < self.eval_episode_max_length or self.eval_episode_max_length == 0)

    def make_probability_map(self, converted_samples):
        """
        Determines the probability of each sample for each sampler
        :param converted_samples:
        :return:
        """
        assert isinstance(self.sampler, BOSS.BOSS), 'Only BOSS needs this'
        sampler_probability_map = {}
        for sampler, obj in self.sampler.sampler_object_dict.items():
            sampler_probability_map[sampler] = []
            for episode in converted_samples:
                for ind, sample in enumerate(episode):
                    sample_probability = obj.get_sample_probability(sample)
                    sampler_probability_map[sampler].append(sample_probability)

        return sampler_probability_map

    def _get_num_samples(self, samples, num_episodes):
        """
        Finds the number if samples in the first n episodes
        :param samples: list of all samples collected
        :param num_episodes: number of episodes to test on
        :return:
        """
        t = 0
        s = 0
        for ind in samples[:num_episodes]:
            t += ind.time
        if isinstance(self.sampler, PolledSampler):
            for ind in self.all_derived_samples[:num_episodes]:
                s += len(ind)

        print('num samples:', t, 'derived samples', s)
        self.lens.append(t)

    def _write_weights(self, sampler_name, weights_dict, derived_weights_dict):
        """
        Writes weights to a txt file
        :return:
        """
        f = open(sampler_name + '.txt', 'w')
        weight_list = weights_dict[sampler_name] + derived_weights_dict[sampler_name]
        for i in weights_dict[sampler_name]:
            f.write(str(i) + ' ')

        f.close()
        w = self.asys.policy_groups[0].policy.function_approximator.weights
        f = open(sampler_name + '_policy_weights' + '.txt', 'w')
        for i in w:
            f.write(str(i) + ' ')

        f.close()
        return weight_list

    def _set_boss_sampler(self, max_sampler_obj, max_sampler_name):
        """
        Updates boss sampler
        """
        if max_sampler_obj:
            self.sampler.current_sampler_name = max_sampler_name
            self.sampler.current_sampler = max_sampler_obj
            self.sampler_schedule_arr.append(max_sampler_name)
            print('Selected:', max_sampler_name)
            self.sampler.sampler_occurance_dict[max_sampler_name] += 1
        else:
            raise RuntimeError("No Sampler to Select")

    def _sa_ind_all_samples(self, all_samples):
        """
        Evaluate all samples into the s, a, s' samples with basis function states
        :param all_samples:
        :return:
        """
        start = time.time()
        sa_ind_samples = np.asarray(self.eval_all_samples(all_samples), dtype=object)
        sa_ind_derived_samples = np.asarray(self.eval_all_derived_samples(self.derived_samples_per_sample),
                                            dtype=object)
        time_to_convert = time.time() - start
        return sa_ind_samples, sa_ind_derived_samples, time_to_convert

    def eval_and_time_boss(self, sa_ind_samples, sa_ind_derived_samples, sa_ind_samples_per_sampler,
                           sa_ind_derived_samples_per_sampler, sampler="BOSS", num_episodes=3):
        """
        Evaluate all samples into the s, a, s' samples with basis function states
        :return:
        """
        start_time2 = time.perf_counter()
        reward = self.eval_policy_boss(sa_ind_samples, sa_ind_derived_samples, sa_ind_samples_per_sampler,
                                  sa_ind_derived_samples_per_sampler=sa_ind_derived_samples_per_sampler,
                                  sampler=sampler, num_episodes=num_episodes, use_eval_environment=True)
        time_to_subtract = (time.perf_counter() - start_time2)
        return reward, time_to_subtract

    def _append_kl_divergence(self, num_samples):
        """
        Appends KL direvergence to the kl_divergence lists
        """
        target_dist = copy.deepcopy(self.sampler.target_distribution)
        current_dist = self.sampler.get_distribution()
        self.entropies.append(self.entropy(np.asarray(current_dist)))
        self.target_entropy = self.entropy(np.asarray(target_dist))
        self.kl_divergences.append(self.kl_divergence(np.asarray(target_dist), np.asarray(current_dist)))
        self.kl_lens.append(num_samples)

    def _get_derived_samples_per_sample(self):
        """
        Gets a list of lists of the derived samples collected for each sampler
        :return:
        """
        if self.collect_inhibited or self.collect_abstract:
            self.derived_samples_per_sample = self.sampler.get_per_sample_derived()
        else:
            self.derived_samples_per_sample = []

    def _update_novel_states_count(self, converted_samples):
        """
        used to track the number of novel states encountered as a function of samles collected
        """
        if self.config.novel_states_count:
            if not self.derived_samples_per_sample:
                self.novel_sa_visits = self._find_novel_sa_visits(self._create_single_list(converted_samples))
            else:
                print('SA visits with derived!')
                self.novel_sa_visits = self._find_novel_sa_visits_with_derived(
                    self._create_single_list(converted_samples))
        else:
            self.novel_sa_visits = []

    def _get_samples(self, conv_samples, num_samples, fraction=.5):
        """
        gets the list of samples including derived limited to num_samples
        :return:
        """
        if fraction > 1 or fraction<0:
            raise ValueError('Fraction cannot be greater than 1 or less than 0')

        # Get derived samples
        if (self.collect_abstract or self.collect_inhibited) and fraction>0:
            derived_samples = self._remove_redundant_samples(self.all_derived_samples[:num_samples], fraction=fraction)
            all_samples = conv_samples + derived_samples
        else:
            all_samples = conv_samples

        return all_samples

    def _get_all_samples(self, conv_samples, fraction=.5):
        """
        Gets all samples incuding derived
        :return:
        """
        # Get derived samples
        if (isinstance(self.sampler, PolledSampler) or isinstance(self.sampler, PolledSampler)) and fraction > 0:
            derived_samples = self._remove_redundant_samples(self.all_derived_samples, fraction=fraction)
            all_samples = conv_samples + derived_samples
        else:
            all_samples = conv_samples

        all_samples = self._create_single_list(all_samples)
        return all_samples

    def _get_all_derived_samples(self):
        # Get derived samples
        if self.collect_inhibited or self.collect_abstract:
            return self._create_single_list(self.all_derived_samples)
        else:
            return []

    def _append_all(self, rew, rew_arr, rt, rt_arr, obj=None, obj_arr=None):
        """
        Appends rewards, runtimes and objective function values to their respective arrays
        """
        rew_arr.append(rew)
        if obj and not obj_arr is None: obj_arr.append(obj)
        rt_arr.append(rt)

    def _update_derived(self, last_obs, done, save = True, append=True):
        # Includes inhibited_abstract samples
        inhibited_samples, abstract_samples = self.sampler.get_derived_samples(last_obs, done)
        if save:
            if self.collect_inhibited and self.collect_abstract:
                self.all_derived_samples.append(inhibited_samples + abstract_samples)
            elif self.collect_inhibited:
                self.all_derived_samples.append(inhibited_samples)
            else:
                self.all_derived_samples.append(abstract_samples)

        return len(inhibited_samples) + len(abstract_samples)

    def _save_traj(self, episode_trajectory):
        if self.save_traj:
            e_traj = episode_trajectory.get_agent_trajectories()[0]
            self.all_traj.extend([int(i) for i in e_traj.actions])

    def _append_trajectories(self, episode_trajectory, last_obs, done):
        self.sample_lens.append(episode_trajectory.time)
        self.episode_trajectories.append(episode_trajectory)
        self.episode_trajectories = self.episode_trajectories[-self.num_trajectories:]
        self.last_obs.append(last_obs)
        self.sucessful_traj.append(done)

    def _delete_after_run(self):
        del self.episode_trajectories
        del self.sampler
        del self.all_derived_samples

    def _find_novel_sa_visits(self, samples):
        novel_states_count = 0
        novel_states_visited = []
        policy = self.asys.policy_groups[0].policy
        domain_obs = policy.domain_obs
        states_array = np.zeros(shape=(self.basis.size()))

        for sample in samples:
            state = np.asarray(self._map_table_indices(sample[0], domain_obs))
            action = sample[1]
            state_index = self.basis.get_state_action_index(state, action)

            # Check if state, action is novel
            if states_array[state_index] == 0:
                novel_states_count += 1
                states_array[state_index] = 1

            novel_states_visited.append(novel_states_count)

        return novel_states_visited

    def _find_novel_sa_visits_with_derived(self, samples):
        novel_states_count = 0
        novel_states_visited = []
        policy = self.asys.policy_groups[0].policy
        domain_obs = policy.domain_obs
        states_array = np.zeros(shape=(self.basis.size()))

        for curr_index, sample in enumerate(samples):
            derived_samples = self.derived_samples_per_sample[curr_index]
            all_samples = [sample] + derived_samples
            for s in all_samples:
                state = np.asarray(self._map_table_indices(s[0], domain_obs))
                action = s[1]
                state_index = self.basis.get_state_action_index(state, action)

                # Check if state, action is novel
                if states_array[state_index] == 0:
                    novel_states_count += 1
                    states_array[state_index] = 1

            novel_states_visited.append(novel_states_count)

        return novel_states_visited

    def _init_basis(self):
        policy = self.asys.policy_groups[0].policy
        domain_obs = policy.domain_obs
        act_domain = policy.domain_act

        # get number of states
        num_states = [domain_item.num_values() for domain_item in domain_obs.items]
        num_actions = act_domain.full_range
        self.basis = ExactBasis.ExactBasis(np.asarray(num_states), num_actions)

    def append_ind_list(self, sampler_start):
        if self.sampler.current_sampler_name in self.sampler_ind_dict:
            self.sampler_ind_dict[self.sampler.current_sampler_name].append((sampler_start,
                                                                             self.episode_step))

        else:
            self.sampler_ind_dict[self.sampler.current_sampler_name] = [(sampler_start,
                                                                         self.episode_step)]

        return sampler_start

    def _remove_redundant_samples(self, samples, fraction=.5, limit=10, use_limit=False):
        """
        Removes some of the samples that are redundant
        Will either keep a fraction of the redundant samples or limit the prevalence
        The goal is to trim as many samples as possible without changing the orignal too much distribution
        :param samples: samples to trim
        :param fraction: fraction of redundant samples to trim
        :param limit: numeric limit of redundant samples to keep (NOT GOOD TO USE)
        :param use_limit: Flag to use limit
        :return: Trimmed samples
        """
        if fraction == 1.0:
            return samples
        reduced_samples = []
        policy = self.asys.policy_groups[0].policy
        domain_obs = policy.domain_obs
        act_domain = policy.domain_act

        # get number of states
        num_states = [domain_item.num_values() for domain_item in domain_obs.items]
        num_actions = act_domain.full_range

        # Add one for no s_prime (goal states)
        states_array = np.zeros(shape=(np.prod(num_states) * num_actions, np.prod(num_states) + 1))

        s_a_dist = np.zeros(shape=(np.prod(num_states) * num_actions))
        s_prime_dist = np.zeros(shape=np.prod(num_states) + 1)

        start = time.time()
        samples_sum = 0
        nr_samples = []

        # Get Original distribution
        for episode in samples:
            for sample in episode:
                state_index = sample[0]
                try:
                    s_prime_index = sample[-1][0] if not len(sample[-1]) == 0 else -1
                except:
                    warnings.warn("Error encountered while trimming samples, returning original sample list")
                    return samples

                # Append samples that have not been seen before
                if states_array[state_index][s_prime_index] == 0:
                    nr_samples.append((copy.deepcopy(sample), state_index, s_prime_index))

                samples_sum += 1
                states_array[state_index][s_prime_index] += 1
                s_a_dist[state_index] += 1
                s_prime_dist[s_prime_index] += 1

        keep_count = 0
        new_s_a_dist = np.zeros(shape=len(s_a_dist))
        new_s_prime_dist = np.zeros(shape=len(s_prime_dist))

        for state in nr_samples:

            # get occurances using s_a, s'
            occurances = states_array[state[1]][state[-1]]

            if use_limit:
                keep = int(min(occurances, limit))
            else:
                # Or just round up
                keep = int(max(1, round(occurances * fraction)))

            keep_count += keep
            new_s_a_dist[state[1]] += keep
            new_s_prime_dist[state[-1]] += keep

            for i in range(0, keep):
                reduced_samples.append(copy.deepcopy(state[0]))

        self.kept.append(keep_count)

        # Get new SA distribution
        new_s_a_dist /= keep_count
        new_s_prime_dist /= keep_count

        # Get old SA Distribution
        s_a_dist /= samples_sum
        s_prime_dist /= samples_sum

        self.sa_kl.append(self.kl_divergence(new_s_a_dist, s_a_dist))
        self.sprime_kl.append(self.kl_divergence(new_s_prime_dist, s_prime_dist))

        print('kept:', keep_count, 'Samples', time.time() - start)
        return [reduced_samples]

    def _display_optimizer_arrays(self, obj, rt, rew, next_x=None, keep_x=None, fractions=None):
        """
        Displays arrays that are used in the fraction optimization algorithm
        :return:
        """
        if next_x and keep_x: print('Next:', next_x, 'Keep', keep_x)
        if fractions: print("fractions:", fractions)
        print('objectives', obj)
        print('Runtimes', rt)
        print('rew', rew)

    def run_eval_samples_boss(self, sa_ind_samples, sa_ind_derived_samples, sa_ind_samples_per_sampler,
                        sa_ind_derived_samples_per_sampler=None, sampler=None, num_episodes=25, use_eval_environment=False, agent_id=0):
        """
        Evaluates the current policy on 100 exploit episodes
        :return:
        """
        rews = 0
        time = 0
        not_added_index = 0
        done = True
        for episode_num in range(num_episodes):
            self.episode_num = episode_num
            self.flags.exploit = True
            self.flags.learn = False
            self.before_episode()
            episode_trajectory = self.run_episode(use_eval_environment=use_eval_environment)
            episode_trajectory.cull()

            if sampler.lower() != "boss":
                observation_request = self.sampler.observe_request()
                if use_eval_environment:
                    last_obs = self.evaluation_environment.observe(observation_request)
                else:
                    last_obs = self.env.observe(observation_request)

                converted_traj = self.convert_trajectory([episode_trajectory], last_obs=last_obs)[0].tolist()

                mapped_s = self.map_all([converted_traj])[0]
                evaluated_samples = self.eval_all_samples(mapped_s)

                # self._check_sampler_distribution_macro(mapped_s)
                sa_ind_samples.extend(evaluated_samples)

                # if 'boss' not in sampler.lower():
                #     if sampler not in sa_ind_samples_per_sampler:
                #         sa_ind_samples_per_sampler[sampler] = evaluated_samples
                #     else:
                #         sa_ind_samples_per_sampler[sampler].extend(evaluated_samples)

                observation_list = episode_trajectory.get_agent_trajectory(agent_id).observations.tolist()
                observation_list.append(list(last_obs[agent_id]))

                all_inhibited_s = []
                all_abstracted_s = []

                ## GENERATE Inhibited samples
                if self.sampler.collect_inhibited and self.sampler.collect_abstract:
                    for ind, observation in enumerate(observation_list):
                        if ind == len(observation_list) - 1:
                            continue

                        inhibited_s_arr = self.generate_inhibited_samples(observation, agent_id, observation_list[ind + 1])
                        all_inhibited_s.append(inhibited_s_arr)

                        primitive_action = mapped_s[ind][1]
                        actual_action = self.get_action(primitive_action)
                        policy_group = self.sampler.hierarchical_policy_dict[agent_id][actual_action]
                        abstracted_s_arr = self.generate_abstract_samples(observation,
                                                                          policy_group,
                                                                          agent_id,
                                                                          primitive_action,
                                                                          observation_list[ind + 1],
                                                                          -1)
                        all_abstracted_s.append(abstracted_s_arr)

                    sa_inhibited_derived = self.eval_all_derived_samples(all_inhibited_s)
                    sa_abstract_derived = self.eval_all_derived_samples(all_abstracted_s)
                    sa_derived_inhibited_single = self._create_single_list(sa_inhibited_derived)
                    sa_derived_abstract_single = self._create_single_list(sa_abstract_derived)

                    sa_ind_derived_samples.extend(sa_derived_inhibited_single + sa_derived_abstract_single)

            rews += episode_trajectory.get_agent_total_rewards()[0]
            time += episode_trajectory.time
        return rews, time

    def run_eval_samples(self, num_episodes=25, use_eval_environment=False, agent_id=0):
        """
        Evaluates the current policy on 100 exploit episodes
        :return:
        """
        rews = 0
        time = 0
        not_added_index = 0
        done = True
        for episode_num in range(num_episodes):
            self.episode_num = episode_num
            self.flags.exploit = True
            self.flags.learn = False
            self.before_episode()
            episode_trajectory = self.run_episode(use_eval_environment=use_eval_environment)
            episode_trajectory.cull()
            rews += episode_trajectory.get_agent_total_rewards()[0]
            time += episode_trajectory.time
        return rews, time

    def extend_sa_derived_samples(self, single_episode_derived, sa_ind_derived_samples_per_sampler,
                                  sa_ind_derived_samples, not_added_index, single_weights_dict=None,
                                  derived_weights_dict=None):
        """
        Append the samples collected to their respctive sampler in the boss sampler dictionaries
        :param single_episode_derived: the sampls derived in the last episode
        :param sa_ind_derived_samples_per_sampler: converted samples in dictionary for each sampler
        :param sa_ind_samples: list of collected samples that have been converted to basis function s, s'
        :param not_added_index: the index of samples that have not been added to the sa_ind arrays
        :param single_weights_dict:
        :param derived_weights_dict: dict of weights of derived samples
        :return: sa_ind_derived_samples with added new samples
        """
        if self.use_weights:
            temp_sa_der_samples, temp_derived_weights_dict = self.eval_all_derived_samples(
                single_episode_derived,
                weights=single_weights_dict)
        else:
            temp_sa_der_samples = self.eval_all_derived_samples(single_episode_derived)

        for k, v in self.sampler_ind_dict.items():
            if not sa_ind_derived_samples_per_sampler or k not in sa_ind_derived_samples_per_sampler:
                sa_ind_derived_samples_per_sampler[k] = []

            for slice in v:
                if slice[0] >= not_added_index:
                    sa_ind_derived_samples_per_sampler[k].extend(
                        self._create_single_list(temp_sa_der_samples[slice[0]:slice[1]]))

        sa_ind_derived_samples.extend(self._create_single_list(temp_sa_der_samples[not_added_index:]))

        if self.use_weights:
            for k, v in temp_derived_weights_dict.items():
                derived_weights_dict[k].extend(v)

            return sa_ind_derived_samples, derived_weights_dict
        return sa_ind_derived_samples

    def extend_sa_samples(self, sa_ind_samples_per_sampler, not_added_index, mapped_s, evaluated_samples):
        """
        Append the samples collected to their respctive sampler in the boss sampler dictionaries

        :param sa_ind_samples_per_sampler: a dictionary mapping each sampler to its list of collected samples
        :param not_added_index:
        :param mapped_s: samples that have been converted to table index
        :param evaluated_samples: samples that have been mapped to the basis function
        :return:
        """
        for k, v in self.sampler_ind_dict.items():
            if k not in sa_ind_samples_per_sampler:
                sa_ind_samples_per_sampler[k] = []

            for slice in v:
                if slice[0] >= not_added_index:
                    if self.sampler.check_dist:
                        self._check_sampler_distribution_macro_2(not_added_index, mapped_s, k, slice)
                    sa_ind_samples_per_sampler[k].extend(
                        evaluated_samples[slice[0] - not_added_index:slice[1] - not_added_index])

        return sa_ind_samples_per_sampler

    def map_all(self, converted_samples):
        """
        Map all samples from native representation to table index.
        """
        policy = self.asys.policy_groups[0].policy
        domain_obs = policy.domain_obs

        for ep_num, ep in enumerate(converted_samples):
            for ind, i in enumerate(ep):
                if i[0] is not None:
                    try:
                        converted_samples[ep_num][ind][0] = self._map_table_indices(i[0], domain_obs)
                    except:
                        print(i[0])
                if i[-1] is not None:
                    converted_samples[ep_num][ind][-1] = self._map_table_indices(i[-1], domain_obs)

        return converted_samples

    def initialize_episode(self, num_samples, episode_num):
        self.episode_num = episode_num
        self.flags.exploit = False
        self.flags.learn = False
        self.before_episode()
        self.episode_max_length = min(self.episode_max_length, self.samples_target - num_samples)

    def continue_running(self, num_samples, sample_lens, episode_num, orig_samples_len):
        """
        Contains the criteria for continuing an offline learning analysis
        :param num_samples:
        :param sample_lens:
        :param episode_num:
        :param orig_samples_len:
        :return:
        """
        return (self.samples_target > num_samples >= 0 and len(sample_lens) > 0) or \
                (self.samples_target < 0 and episode_num <= self.episodes_to_run) or \
                (self.samples_target > num_samples >= 0 and orig_samples_len == [])

    def abstract_all_observation_domains(self, hierarchy):
        h_obs_domain = {}
        for action in hierarchy.actions:
            if not hierarchy.actions[action]['primitive']:
                if self.config.environment.name == "BitFlip" or self.config.environment.name == "BitFlipMiddle":
                    h_obs_domain[action] = self.env.abstracted_observation_domain(hierarchy.actions[action]['state_variables'],
                                                                                 action)
                else:
                    h_obs_domain[action] = self.env.abstracted_observation_domain(hierarchy.actions[action]['state_variables'])

        return h_obs_domain

    def _display_arrays(self):
        """
        Displays all of the arrays holding important information:
        Number of actual samples, evaluative reward, runtime, number of derived samples kept,
        The KL divergence between the S,A distributions pre and post prunning
        The KL divergence between the S' distributions pre and post prunning
        :return:
        """
        print('lens:', self.lens)
        print('rew:', self.rewards)
        print('rt', self.rt)
        print('Derived:', self.derived_samples)

    def _append_multiple(self, large_obj, small_obj, large_rew, small_rew, obj, rew):
        """
        appends multiple objective function values and reward values an array
        """
        obj.append(large_obj)
        obj.append(small_obj)
        rew.append(large_rew)
        rew.append(small_rew)

    def _calculate_objective_function(self, reward, rt):
        """
        The objective function for optimizing the runtime/reward tradeoff
        :param reward: average eval reward
        :param rt: runtime
        :return: objective function value
        """
        return .1 * reward - rt

    def _get_obj(self, conv_samples, fraction, i):
        """
        Finds the objective function value after evaluating the current policy
        :param conv_samples:
        :param fraction:
        :param i:
        :return:
        """
        samples = self._get_samples(conv_samples[:i], i, fraction=fraction)
        rew, rt = self.eval_policy2(samples)
        obj = self._calculate_objective_function(rew, rt)
        return rew, rt, obj

    def _map_table_indices(self, states: np.ndarray, domain_obs) -> Tuple[int, ...]:
        """
        Map a single state from native representation to table index.
        :param states: State in native representation
        :return: Table index
        """
        mapped_state = []
        for feature in domain_obs.items:
            # Get individual domain item state and map it to the requested interpretation
            domain_state = domain_obs.get_item_view_by_item(states, feature)
            domain_mapped_state = FeatureConversions.as_index(feature, domain_state)
            mapped_state.append(domain_mapped_state)
        return np.asarray(mapped_state)

    def _plot_obj(self, converted_samples):
        """
        Runs the learning algorithm on many different fractions to analyze the shape of the objective function
        :param converted_samples: Samples that have been converted to a list in form [[s1, a1, r1, s1'], ...]
        """
        sample_lens = self.config.eval_samples
        rew, rt, obj = ([] for i in range(3))
        fractions = [1.0, .9, .8, .7, .6, .5, .4, .35, .3, .25, .2, .15, .1, .05, .04, .03, .02, .01, 0]
        for i in sample_lens:
            for fraction in fractions:
                new_x_rew, new_x_rt, new_x_obj = self._get_obj(converted_samples, fraction, i)
                self._append_all(new_x_rew, rew, new_x_rt, rt, new_x_obj, obj)
                self._display_optimizer_arrays(obj, rt, rew)

    def _create_single_list(self, all_samples):
        all_samples_list = []
        for episode_samples in all_samples:
            for sample in episode_samples:
                all_samples_list.append(sample)

        return all_samples_list

    def load_samples(self):
        """
        Load previously saved samples
        :return: actual samples, derived samples are saved
        """
        os.chdir(self.name)
        sa_ind_derived_samples = []
        if self.collect_inhibited or self.collect_abstract:
            with np.load(self.name + '_derived_samples_' + str(self.iteration) + '.npy', allow_pickle=True) as data:
                sa_ind_derived_samples = data['arr_0.npy']

        # Load other samples
        with np.load(self.name + '_samples_' + str(self.iteration) + '.npy', allow_pickle=True) as data:
            sa_ind_samples = data['arr_0.npy']

        print('Successfully loaded converted samples: ', self.name + '_samples_' + str(self.iteration))
        os.chdir('..')
        return sa_ind_samples, sa_ind_derived_samples

    def save_samples(self, sa_ind_samples, sa_ind_derived_samples):
        """
        Saves the samples for future use
        :param samples
        """
        if self.iteration == 0:
            try:
                os.mkdir(self.dir_name)
            except OSError:
                pass

        s = True
        try:
            os.chdir(self.dir_name)
        except OSError:
            s = False
            pass

        if self.collect_inhibited or self.collect_abstract:
            file_name = self.name + '_derived_samples_' + str(self.iteration) + '.npy'
            success = True
            try:
                with open(file_name, 'wb') as derived_samples_file:
                    np.savez_compressed(derived_samples_file, sa_ind_derived_samples)
            except OSError:
                print("Unexpected error:", sys.exc_info()[0])
                print('Could not save derived samples')
                success = False
                pass

            if success:
                print('Sucessfully saved:', file_name)
        success = True

        try:
            # Save other samples
            with open(self.name + '_samples_' + str(self.iteration) + '.npy', 'wb') as samples_file:
                np.savez_compressed(samples_file, sa_ind_samples)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            print('Could not save samples')
            pass
            success = False

        if success:
            print('Sucessfully saved:', self.name + '_samples_' + str(self.iteration))

        if s:
            os.chdir('..')

    def _reset_policy_weights(self):
        """
        resets policy for next learning opportunity
        :return:
        """
        for pg in self.asys.policy_groups:
            pg.policy.function_approximator.reset_weights()

    def _check_sample_distribution(self, samples):
        sample_dict = {}

        for i in samples:
            a = i[1][0]
            if a not in sample_dict:
                sample_dict[a] = 1
            else:
                sample_dict[a] += 1

        return sample_dict

    def _check_sampler_distribution_macro_2(self, previous_stoppage, mapped_s, k, slice):
        """
        helper function for check distribution
        """

        x = self._check_sample_distribution(
            mapped_s[slice[0] - previous_stoppage:slice[1] - previous_stoppage])
        if k not in self.sampler.sampler_dist_dict:
            self.sampler.sampler_dist_dict[k] = x

        else:
            for k_1, v_1 in x.items():
                if k_1 in self.sampler.sampler_dist_dict[k]:
                    # print('Added K2', k)
                    self.sampler.sampler_dist_dict[k][k_1] += v_1
                else:
                    # print('Added K3', k)
                    self.sampler.sampler_dist_dict[k][k_1] = v_1


    def _check_sampler_distribution_macro(self, mapped_s):
        """
        helper function for check distribution
        """
        if self.sampler.check_dist:
            k = self.sampler_tag
            x = self._check_sample_distribution(mapped_s)
            if k not in self.sampler.sampler_dist_dict:
                self.sampler.sampler_dist_dict[k] = x

            else:
                for k_1, v_1 in x.items():
                    if k_1 in self.sampler.sampler_dist_dict[k]:
                        self.sampler.sampler_dist_dict[k][k_1] += v_1
                    else:
                        self.sampler.sampler_dist_dict[k][k_1] = v_1

    def _write_sample_distribution(self, iteration=1):

        if (self.config.sampler.name == "BOSS" or self.is_etc) and self.sampler.check_dist:
            with open("sample_dist" + str(iteration) + ".txt", 'w') as f:
                for key, value in self.sampler.sampler_dist_dict.items():
                    f.write('%s:%s\n' % (key, value))
            with open("sample_dist" + str(iteration) + ".pkl", 'wb') as f:
                pkl.dump(self.sampler.sampler_dist_dict, f)

    @staticmethod
    def kl_divergence(vec1: np.ndarray, vec2: np.ndarray):
        """
        Calculates the kl divergence of two vectors, lower numbers suggests distributions are similar
        Must be a probability distribution
        :return: kl_divergence
        """
        if abs(1 - np.sum(vec1)) > 1e-3 or abs(1 - np.sum(vec2)) > 1e-3:
            raise ValueError('Must be a probability distribution')

        # Add a small amount to avoid a divide by zero error
        vec2 += 1e-9
        vec1 += 1e-9
        return np.sum(vec1 * np.log(vec1 / vec2), 0)

    @staticmethod
    def entropy(vec1: np.ndarray):
        """
        Calculates the kl divergence of two vectors, lower numbers suggests distributions are similar
        Must be a probability distribution
        :return: kl_divergence
        """
        if abs(1 - np.sum(vec1)) > 1e-3:
            raise ValueError('Must be a probability distribution')

        return sp.stats.entropy(vec1)

    def _fraction_tuner(self, converted_samples, samples, starting_fraction=.5):
        """
        Attempts to find the optimal fraction of derived samples to keep without negatively impacting runtime
        Can be prone to over-trimming and can prune overly aggressively

        :param converted_samples: Samples that have been converted to a list (see convert_trajectory)
        :param samples: Raw list of samples, just included for running on the specified number of episodes
        :param starting_fraction: Fraction to begin trimming with, will start with 1 and .5 (reccomended)
        """
        iteration_count = 0
        curr_fract = starting_fraction
        fraction_array = []

        # These ratios are selected so that a decreasing and then increasing will result in no net change
        dec_rat = .75
        inc_rat = 1.33

        # This determines after how many episodes to stop and find and eval policy
        sample_lens = self.config.eval_samples
        for i in sample_lens:
            self._get_num_samples(samples, i)
            start = time.time()

            # Calculate reward with all samples and the current_fraction * derived samples
            if iteration_count == 0:

                rew_all = self.eval_policy()
                rew_half = self.eval_policy()
                self.rewards.append(max(rew_all, rew_half))

                # If difference is less than 10 percent, move fraction lower
                if rew_all - rew_half < .1 * rew_all:
                    curr_fract *= dec_rat
                    fraction_array.append(.5)
                else:
                    curr_fract *= inc_rat
                    fraction_array.append(1.0)
            else:
                fraction_array.append(curr_fract)
                all_samples = self._get_samples(converted_samples[:i], i, fraction=curr_fract)
                self.rewards.append(self.eval_policy(25))

                # Move fraction in the direction of maximum reward
                curr_fract = curr_fract*dec_rat if self.rewards[-1] > self.rewards[-2] else curr_fract*inc_rat
            iteration_count += 1

            self.rt.append(time.time() - start)
            self._reset_policy_weights()

            self._display_arrays()
            print(fraction_array)

    def _objecive_optimizer(self, converted_samples):
        """
        Finds the fraction of derived samples to keep that maximizes the objective function
        Designed to be used on a single amount of samples
        :param converted_samples: Samples that have been converted to a list in form [[s1, a1, r1, s1'], ...]
        :return: prints all results
        """
        sample_lens = self.config.eval_samples

        big_x = 1.0
        small_x = 0.5
        iterations_count = 0
        diff = .125

        objective_functions = []
        fractions = [1, .5]
        rew = []
        rt = []
        for i in sample_lens:
            big_x_rew, big_x_rt, big_x_obj = self._get_obj(converted_samples, big_x, i)
            rt.append(big_x_rt)

            small_x_rew, small_x_rt, small_x_obj = self._get_obj(converted_samples, small_x, i)
            rt.append(small_x_rt)

            # Keep larger objective function
            keep_x_obj = small_x_obj if small_x_obj > big_x_obj else big_x_obj

            # Keep fraction that had the higher objective function
            keep_x = small_x if small_x_obj>big_x_obj else big_x

            # Scale for next iteration based on highest objective function
            next_x = small_x*.5 if small_x_obj>big_x_obj else small_x * 1.5

            self._append_multiple(big_x_obj, small_x_obj, big_x_rew, small_x_rew, objective_functions, rew)

            # Continue search for up to 10 iterations, when the algo has converged
            while iterations_count<10:

                # Calculate objective function with new fraction
                new_x_rew, new_x_rt, new_x_obj = self._get_obj(converted_samples, next_x, i)
                self._append_all(new_x_rew, rew, new_x_rt, rt, new_x_obj, objective_functions)
                fractions.append(next_x)

                # If new fraction out performed old fraction
                if new_x_obj > keep_x_obj:
                    keep_x = next_x
                    keep_x_obj = new_x_obj

                    # Move the Next X in the direction of improved performance (i.e raise if larger fraction was better)
                    next_x = keep_x + diff if next_x > keep_x else keep_x - diff
                else:
                    next_x = next_x - diff if next_x > keep_x else next_x + diff

                diff = diff/2
                iterations_count += 1
                self._display_optimizer_arrays(objective_functions, rt, rew, next_x, keep_x, fractions)

    def save_flattened_hierarchy(self, dir_name, save_name, config):
        # Save flattened hierarchy for analysis
        try:
            os.chdir(dir_name)
            with open(save_name + "_flattened_hierarchy.json", "w") as outfile:
                json.dump(config.environment.action_hierarchy, outfile, indent=2)
            os.chdir('..')
            print('Saved Flattened Hierarchy:', save_name + "_flattened_hierarchy.json")
        except Exception as e:
            print(e, 'Could not save hierarchy')
            pass

    def _objective_tuner(self, converted_samples, samples, starting_fraction=.5):
        """
        Finds the fraction of derived samples to keep that maximizes the objective function
        Designed to be used on multiple sample lengths and will optimize the fraction as more samples are added
        This algorithm is susceptible to local minima due to overtrimming before the optimal policy could have
        possibly be reached.
        :param converted_samples: Samples that have been converted to a list in form [[s1, a1, r1, s1'], ...]
        :param samples: original list of samples used for finding number of samples to evaluate on
        :param starting_fraction: fraction of samples to initially keep. The algo with start by comparing this value
        to keeping all samples, so a natural choice is 0.5
        :return:
        """
        iteration_count = 0
        current_fraction = starting_fraction
        fraction_array = []
        obj = []
        diff = .125
        # This determines after how many episodes to stop and find and eval policy
        sample_lens = self.config.eval_samples
        for i in sample_lens:
            self._get_num_samples(samples, i)
            start = time.time()

            if iteration_count == 0:
                # Find the objective function values of first two fractions (1.0 and .5 by default)
                rew_all, rt_all, all_obj = self._get_obj(converted_samples, 1.0, i)
                rew_half, rt_half, half_obj = self._get_obj(converted_samples, current_fraction, i)

                self.rewards.append(max(rew_all, rew_half))
                obj.append(max(all_obj, half_obj))

                # If difference is less than 10 percent, move fraction lower
                if half_obj>all_obj:
                    current_fraction = .25
                    fraction_array.append(current_fraction)
                # Move fraction higher
                else:
                    current_fraction = .75
                    fraction_array.append(1.0)

            else:
                fraction_array.append(current_fraction)
                rew_t, rt_t, obj_t = self._get_obj(converted_samples, current_fraction, i)

                self.rewards.append(rew_t)
                obj.append(obj_t)

                # Move fraction in direction of best performance
                if obj_t > obj[-2]:
                    current_fraction -= diff
                else:
                    current_fraction += diff
                diff = diff/2

            iteration_count += 1

            self.rt.append(time.time() - start)
            self._reset_policy_weights()
            self._display_arrays()
            print(fraction_array)

    def generate_inhibited_samples(self, observation, agent_id, s_prime):
        """
        Generates inhibited action samples, or samples that set the reward of taking an illegal action (from hierarchy)
        to a large negative number

        Only does it for the current observation, not for a list of samples as proposed in Devin's thesis

        :param observation: current observation
        :param agent_id: current agent_id
        :return:
        """
        # root_pg = self.sampler.completion_function_pg[agent_id]['Root']
        inhibited_s_arr = []

        policy = self.asys.policy_groups[0].policy
        domain_obs = policy.domain_obs
        mapped_obs = self._map_table_indices(observation, domain_obs)
        mapped_s_prime = self._map_table_indices(s_prime, domain_obs)

        inhibited_index = self.sampler.inhibited_actions_basis.get_state_action_index(mapped_obs, 0)

        # Uses the previously found non-reachable primitives if this state has been tested
        if self.sampler.inhibited_actions_arr[inhibited_index] is None:
            non_term_prim, reachable_subtasks = self.sampler.get_all_reachable_non_term_actions(observation, "Root",
                                                                                        agent_id)

            term_prim = [i for i in self.sampler.primitive_action_map.keys() if i not in non_term_prim]
            self.sampler.inhibited_actions_arr[inhibited_index] = (term_prim, reachable_subtasks)
        else:
            [term_prim, _] = self.sampler.inhibited_actions_arr[inhibited_index]

        blocked_acts = term_prim

        for inhibited_action in blocked_acts:
            prim_action = self.sampler.primitive_action_map[inhibited_action]
            inhibited_samples = [mapped_obs, np.asarray([prim_action]), self.sampler.min_reward, mapped_s_prime]
            inhibited_s_arr.append(inhibited_samples)

        return inhibited_s_arr

    def get_action(self, primitive_action):
        for k,v in self.sampler.primitive_action_map.items():
            if v == primitive_action:
                return k

        raise ValueError('Cannot find action')

    def generate_abstract_samples(self, observation, policy_group, agent_id, primitive_action, s_prime, reward, current_node="Root"):
        """
        Generates abstract samples by changing the values of state variables that
        the hierarchy determines are irrelevant

        :param observation: Current (full) observation
        :param policy_group: current policy group
        :param agent_id: agent-id for policy group
        :param primitive_action: primitive action of sample
        :return:
        """
        # if self.display_distribution and self.add_derived:
        root_pg = self.sampler.completion_function_pg[agent_id]['Root']
        state_list, s_prime_list = self.sampler.get_irrelevant_states(observation, policy_group, agent_id, s_prime)
        abstract_samples_l = []
        policy = self.asys.policy_groups[0].policy
        domain_obs = policy.domain_obs
        # inhibited_actions_l = []
        for ind, state in enumerate(state_list):
            mapped_obs = self._map_table_indices(state, domain_obs)
            mapped_s_prime = self._map_table_indices(s_prime_list[ind], domain_obs)
            sample = [mapped_obs, np.asarray(primitive_action), reward, mapped_s_prime]
            abstract_samples_l.append(sample)

            # generate inhibited from abstract
            if self.sampler.collect_inhibited and self.sampler.collect_abstract:
                inhibited_index = self.sampler.inhibited_actions_basis.get_state_action_index(mapped_obs, 0)

                # Uses the previously found non-reachable primitives if this state has been tested
                if self.sampler.inhibited_actions_arr[inhibited_index] is None:
                    non_term_prim, reachable_subtasks = self.sampler.get_all_reachable_non_term_actions(state, "Root",
                                                                                                        agent_id)
                    # print(non_term_prim, reachable_subtasks, mapped_obs, inhibited_index)
                    term_prim = [i for i in self.sampler.primitive_action_map.keys() if i not in non_term_prim]
                    self.sampler.inhibited_actions_arr[inhibited_index] = (term_prim, reachable_subtasks)
                else:
                    [term_prim, _] = self.sampler.inhibited_actions_arr[inhibited_index]

                for inhibited_action in term_prim:
                    prim_action = self.sampler.primitive_action_map[inhibited_action]
                    inhibited_samples = [mapped_obs, np.asarray([prim_action]), self.sampler.min_reward, mapped_s_prime]
                    abstract_samples_l.append(inhibited_samples)

        return abstract_samples_l

class ControllerFlags(object):

    def __init__(self):
        self.learn = True
        self.exploit = False
        self.visualize = False
