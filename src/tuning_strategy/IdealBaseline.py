from typing import List, TYPE_CHECKING
import numpy as np
import os
import abc

from config.config import ConfigItemDesc
from .base import TuningStrategy
from algorithm.base import Algorithm
from .utils import generate_param_set, OffPolicyLearner, Logger
from common.trajectory import TrajectoryCollection

if TYPE_CHECKING:
    from config import Config
    from controller import MDPController

class IdealBaseline(TuningStrategy):
    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return TuningStrategy.get_class_config() + [
            ConfigItemDesc(name='ensemble_size',
                           check=lambda s: isinstance(s, int),
                           info='Number of parameter configurations.')
        ]

    def __init__(self, controller: 'MDPController', config: 'Config', pg_id: int):
        super().__init__(controller, config, pg_id)
        self.ensemble_size = self.config.ensemble_size
        self.logger = Logger(self.result_file)

        self.in_update = False
        self.is_online = False
        self.alg_index = None
        self.alg_set = []

    def update_alg(self):
        self.in_update = True
        expected_evaluation_rewards = {}

        for j in range(len(self.alg_set)):
            current_policy = self.alg_set[j].policy.serialize()
            current_model = self.alg_set[j].model

            # Set seed state
            self.controller.set_seed_state(self.traj_seed_state)

            # Train
            self.controller.flags.exploit = False
            self.controller.flags.learn = True

            # Learn one 1 trajectory
            self.set_alg(j, False)
            self.controller.run_episode()

            # Evaluate
            self.controller.flags.exploit = True
            self.controller.flags.learn = False

            # Run evaluation episodes
            rewards = []
            eval_episodes = 10
            for _ in range(eval_episodes):
                self.controller.run_episode()
                traj = self.get_pg().trajectory
                rewards.append(np.sum(traj.rewards))

            # Record result and eliminate learning
            expected_evaluation_rewards[j] = np.mean(rewards)

            self.alg_set[j].policy.load(current_policy)
            self.alg_set[j].model = current_model

        # Choose best algorithm based on peeking ahead
        max_eval_reward = max(expected_evaluation_rewards.values())
        max_indices = [k for k,v in expected_evaluation_rewards.items() if v == max_eval_reward]

        # If current index in tie, choose that
        if not self.alg_index is None and self.alg_index in max_indices:
            chosen_index = self.alg_index
        else:
            # Random tie break
            chosen_index = np.random.choice(max_indices)

        self.in_update = False
        print('Max Evaluation Reward: {}'.format(expected_evaluation_rewards[chosen_index]))
        return chosen_index

    def before_run(self):
        # Draw unique parameter sets
        baseline_alg_config = self.controller.config.algorithm
        baseline_policy_config = self.controller.config.policy
        config_set, _ = generate_param_set(self.ensemble_size, baseline_alg_config, baseline_policy_config)

        # Check if online learning
        self.is_online = getattr(baseline_alg_config, "is_online", None) != None

        # Get global config
        global_config = self.controller.config

        # Initialize algorithms
        if self.tune_policy_sampler_params:
            self.alg_set = [OffPolicyLearner(self.get_pg(), alg_config, policy_config, global_config) for alg_config, policy_config in config_set]
        else:
            self.alg_set = [OffPolicyLearner(self.get_pg(), alg_config, baseline_policy_config, global_config) for alg_config, _ in config_set]

        # Set all policies and models with same values as baseline
        pg = self.get_pg()
        for j in range(self.ensemble_size):
            self.alg_set[j].policy.load(pg.policy.serialize())

        # Reset logger
        self.logger.reset(self.alg_set)

    def before_episode(self):
        # Get seed state
        self.traj_seed_state = self.controller.get_seed_state()
        
        if self.controller.flags.learn:
            # Look ahead and choose algorithm based on evaluation trajectories
            self.alg_index = self.update_alg()
            print('Selected Index: {}'.format(self.alg_index))

            # Update controller
            self.set_alg(self.alg_index)

            # Set seed state
            self.controller.set_seed_state(self.traj_seed_state)

            # Reset flags
            self.controller.flags.exploit = False
            self.controller.flags.learn = True

    def on_update(self):
        # Get seed state
        if not self.in_update:
            self.step_seed_state = self.controller.get_seed_state()

    def after_update(self):
        if not self.in_update and self.controller.flags.learn:
            if self.is_online:
                self.update_off_policy_learners(self.step_seed_state)
            else:
                self.update_off_policy_learners(self.traj_seed_state)

    def set_alg(self, alg_index, log=True):
        chosen_alg = self.alg_set[alg_index]
        self.controller.asys.algorithm = chosen_alg.algorithm
        self.controller.asys.policy_groups[self.pg_id].policy.load(chosen_alg.policy.serialize())
        self.controller.asys.policy_groups[self.pg_id].policy.sampler = chosen_alg.policy.sampler
        self.controller.asys.policy_groups[self.pg_id].model = chosen_alg.model

        if log:
            self.logger.log_selection(self.controller.episode_num, self.alg_index)

    def update_off_policy_learners(self, seed_state):
        current_seed_state = self.controller.get_seed_state()

        pg = self.get_pg()
        for alg in self.alg_set:
            # Set seed state
            self.controller.set_seed_state(seed_state)

            alg.learn(pg.trajectory)

        # Set seed state
        self.controller.set_seed_state(current_seed_state)
