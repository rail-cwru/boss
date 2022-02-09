from typing import List, TYPE_CHECKING
import numpy as np
import os
import abc
from copy import deepcopy

from config import ConfigItemDesc, checks, Config
from .base import TuningStrategy
from .utils import generate_param_set, perturb_param_set, draw_param_set
from controller import MDPController

EXPLORE_STRATEGIES = ['perturb', 'resample']

class PBT(TuningStrategy):
    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return TuningStrategy.get_class_config() + [
            ConfigItemDesc(name='ensemble_size',
                           check=checks.positive_integer,
                           info='Number of parameter configurations.'),
            ConfigItemDesc(name='experience_limit',
                           check=checks.positive_integer,
                           info='Minimum experience before exploitation.'),          
            ConfigItemDesc(name='explore_strategy',
                           check=lambda x: x in EXPLORE_STRATEGIES,
                           info='Strategy to explore after expoitation, options: {}.'.format(EXPLORE_STRATEGIES))   
        ]

    def __init__(self, controller: MDPController, config: Config, pg_id: int):
        super().__init__(controller, config, pg_id)
        self.ensemble_size = self.config.ensemble_size
        self.experience_limit = self.config.experience_limit
        self.explore_strategy = self.config.explore_strategy

        self.quantile_count = int(np.ceil(self.ensemble_size * 0.2))
        self.learners = {}
        self.seed_states = {}
        self.experience = {}

    def before_run(self):
        # Draw unique parameter sets
        baseline_alg_config = self.controller.config.algorithm
        baseline_policy_config = self.controller.config.policy

        # Generate config set
        config_set, self.baseline_index = generate_param_set(self.ensemble_size, baseline_alg_config, baseline_policy_config)

        # Get global config
        global_base_config = deepcopy(self.controller.config)

        # Remove parent callback
        global_base_config.callbacks.pop('TuneHyperparameters', None)   

        # Initialize algorithms
        base_seed_state = self.controller.get_seed_state()
        self.learners = {}
        self.seed_states = {}
        self.experience = {i:0 for i in range(self.ensemble_size)}
        for i, configs in enumerate(config_set): 
            
            if i == self.baseline_index:
                continue

            # Create new config
            alg_config, policy_config = configs
            new_config = deepcopy(global_base_config)
            new_config.algorithm = alg_config
            new_config.subconfigs[alg_config.name] = alg_config
            if self.tune_policy_sampler_params:
                new_config.policy = policy_config
                new_config.subconfigs[policy_config.name] = policy_config

            # Include index in output name
            if 'Evaluate' in new_config.callbacks.keys():
                new_config.callbacks['Evaluate'].output_reward_file = '{0}_{2}.{1}'.format(*new_config.callbacks['Evaluate'].output_reward_file.rsplit('.', 1) + [i])
            if 'SaveReward' in new_config.callbacks.keys():
                new_config.callbacks['SaveReward'].file_location = '{0}_{2}.{1}'.format(*new_config.callbacks['SaveReward'].file_location.rsplit('.', 1) + [i])

            # Create new controllers
            self.learners[i] = MDPController(new_config)
            self.seed_states[i] = base_seed_state

        # Set all policies and models with same values as baseline
        pg = self.get_pg()
        for j in range(self.ensemble_size):
            if j != self.baseline_index:
                self.learners[j].asys.policy_groups[self.pg_id].policy.load(pg.policy.serialize())

        directory = os.path.dirname(self.result_file)
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.output_file = '{}_{}.csv'.format(self.result_file, 'eval_experience')
        column_header = ['After Episode', 'Eval Experience']
        with open(self.output_file, mode='w') as f:
            header = ','.join(column_header)
            f.write(header + '\n')

    def exploit_and_explore(self, eval_reward, eval_experience):
        # Identify top and bottom 20%
        lower_quantile = sorted(eval_reward, key=eval_reward.get, reverse=False)[:self.quantile_count]
        upper_quantile = sorted(eval_reward, key=eval_reward.get, reverse=True)[:self.quantile_count]

        unique_upper_quantile = set(upper_quantile) - set(lower_quantile)
        if len(unique_upper_quantile) != self.quantile_count:
            # Repeated reward, choose upper
            remaining_ids = set(eval_reward.keys()) - unique_upper_quantile
            upper_quantile = list(unique_upper_quantile) + list(np.random.choice(list(remaining_ids), self.quantile_count-len(unique_upper_quantile), replace=False))

            # Choose lower
            remaining_ids = remaining_ids - set(upper_quantile)
            lower_quantile = np.random.choice(list(remaining_ids), self.quantile_count, replace=False)
            
        # Perturb lower quantile
        for i in lower_quantile:
            # Randomly choose upper quantile to copy
            copy_index = np.random.choice(upper_quantile)
            if copy_index == self.baseline_index:
                alg_config = deepcopy(self.controller.asys.algorithm.config)
                policy_config = deepcopy(self.controller.asys.policy_groups[self.pg_id].policy.config)
            else:
                alg_config = deepcopy(self.learners[copy_index].asys.algorithm.config)
                policy_config = deepcopy(self.learners[copy_index].asys.policy_groups[self.pg_id].policy.config)

            # Only found the redraw in one document...
            if self.explore_strategy == 'perturb':
                # Perterb
                alg_config, policy_config = perturb_param_set(alg_config, policy_config, self.perturb_factor, self.tune_policy_sampler_params)
            elif self.explore_strategy == 'resample':
                # Redraw
                alg_config, policy_config = draw_param_set(alg_config, policy_config, self.tune_policy_sampler_params)
            else:
                raise Exception('Unknown exploration strategy was specificied: {}'.format(self.explore_strategy))

            # Update configs, recompile policy if required
            if i == self.baseline_index:
                self.controller.asys.algorithm.config = alg_config
                self.controller.asys.policy_groups[self.pg_id].policy.config = policy_config
                self.controller.config.algorithm = alg_config
                self.controller.config.subconfigs[alg_config.name] = alg_config
                self.controller.config.policy = policy_config
                self.controller.config.subconfigs[policy_config.name] = policy_config
            else:
                self.learners[i].asys.algorithm.config = alg_config
                self.learners[i].asys.policy_groups[self.pg_id].policy.config = policy_config
                self.learners[i].config.algorithm = alg_config
                self.learners[i].config.subconfigs[alg_config.name] = alg_config
                self.learners[i].config.policy = policy_config
                self.learners[i].config.subconfigs[policy_config.name] = policy_config

            # Update parameters
            alg_param_names =  [param.name for param in self.controller.asys.algorithm.get_class_config()]
            for p_name in alg_param_names:
                new_value = getattr(alg_config, p_name, None)
                if i == self.baseline_index:
                    setattr(self.controller.asys.algorithm, p_name, new_value)
                else:
                    setattr(self.learners[i].asys.algorithm, p_name, new_value)
            
            policy_sampler_param_names = [param.name for param in self.controller.asys.policy_groups[self.pg_id].policy.sampler.get_class_config()]
            for p_name in policy_sampler_param_names:
                new_value = getattr(policy_config.policy_sampler, p_name, None)
                if i == self.baseline_index:
                    setattr(self.controller.asys.policy_groups[self.pg_id].policy.sampler, p_name, new_value)
                else:
                    setattr(self.learners[i].asys.policy_groups[self.pg_id].policy.sampler, p_name, new_value)

            # Recompile policy if required
            if i == self.baseline_index:
                for pg in self.controller.asys.policy_groups:
                    self.controller.asys.algorithm.compile_policy(pg.policy)
            else:
                for pg in self.learners[i].asys.policy_groups:
                    self.learners[i].asys.algorithm.compile_policy(pg.policy)

            # Finally copy over policy weights
            if copy_index == self.baseline_index:
                self.learners[i].asys.policy_groups[self.pg_id].policy.load(self.controller.asys.policy_groups[self.pg_id].policy.serialize())
            elif i == self.baseline_index:
                self.controller.asys.policy_groups[self.pg_id].policy.load(self.learners[copy_index].asys.policy_groups[self.pg_id].policy.serialize())
            else:
                self.learners[i].asys.policy_groups[self.pg_id].policy.load(self.learners[copy_index].asys.policy_groups[self.pg_id].policy.serialize())

        # Reset experience and traj count
        self.experience = {i:0 for i in range(self.ensemble_size)}

        # Write out the additional eval experience used
        with open(self.output_file, mode='a') as f:
            row = [str(self.controller.episode_num), str(eval_experience)]
            row_str = ','.join(row)
            f.write(row_str + '\n')

    def before_episode(self):
        if all(value >= self.experience_limit for value in self.experience.values()):
            # Grab avg reward from evaluate callback for all other learners
            eval_reward = {}
            eval_experience = 0
            eval_episodes = 10
            for i in range(self.ensemble_size):
                rewards = []
                experiences = []

                if i == self.baseline_index:
                    # Evaluate
                    self.controller.flags.exploit = True
                    self.controller.flags.learn = False

                    # Run evaluation episodes
                    for _ in range(eval_episodes):
                        self.controller.run_episode()
                        traj = self.get_pg().trajectory
                        rewards.append(np.sum(traj.rewards))
                        experiences.append(len(traj))

                    self.controller.flags.exploit = False
                    self.controller.flags.learn = True

                else:
                    # Evaluate
                    self.learners[i].flags.exploit = True
                    self.learners[i].flags.learn = False

                    # Run evaluation episodes
                    for _ in range(eval_episodes):
                        self.learners[i].run_episode()
                        traj = self.learners[i].asys.policy_groups[self.pg_id].trajectory
                        rewards.append(np.sum(traj.rewards))
                        experiences.append(len(traj))

                    self.learners[i].flags.exploit = False
                    self.learners[i].flags.learn = True

                eval_reward[i] = np.mean(rewards)
                eval_experience += np.sum(experiences)
            
            self.exploit_and_explore(eval_reward, eval_experience)

    def after_episode(self):
        # Get experience of baseline 
        learn_traj_experience = len(self.get_pg().trajectory)
        self.experience[self.baseline_index] += learn_traj_experience

        if self.experience[self.baseline_index] >= self.experience_limit:
            seed_state = self.controller.get_seed_state()

            # Run an episode for every learner
            #TODO: Parallelize
            for j in range(self.ensemble_size):
                if j == self.baseline_index:
                    continue

                # Restore seed state
                self.learners[j].set_seed_state(self.seed_states[j])

                while self.experience[j] < self.experience_limit:
                    # Reset flags since after_episode turns them off
                    self.learners[j].flags.exploit = False
                    self.learners[j].flags.learn = True

                    # Run epsiode
                    self.learners[j].episode_num += 1
                    self.learners[j].run_episode()

                    # Retrieve learning trajectory
                    learn_traj_experience = len(self.learners[j].asys.policy_groups[self.pg_id].trajectory)
                    self.experience[j] += learn_traj_experience

                    # Required for evaluate and save reward
                    self.learners[j].after_episode()

                # Save seed state
                self.seed_states[j] = self.learners[j].get_seed_state()

            self.controller.set_seed_state(seed_state)
        
    def after_run(self):
        for j in range(self.ensemble_size):
            if j == self.baseline_index:
                continue

            # Required for evaluate
            self.learners[j].after_run()

    def finalize(self):
        for j in range(self.ensemble_size):
            if j == self.baseline_index:
                continue
            
            # Required for evaluate and save reward
            self.learners[j].finalize()
