from typing import List, TYPE_CHECKING
import numpy as np
import os
import abc

from config.config import ConfigItemDesc
from .base import TuningStrategy
from algorithm.base import Algorithm
from .utils import generate_param_set, OffPolicyLearner, Logger

if TYPE_CHECKING:
    from config import Config
    from controller import MDPController

class FTL(TuningStrategy):
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

        self.is_online = False
        self.alg_index = 0
        self.alg_set = []

    def before_run(self):
        # Draw unique parameter sets
        baseline_alg_config = self.controller.config.algorithm
        baseline_policy_config = self.controller.config.policy
        config_set, _ = generate_param_set(self.ensemble_size, baseline_alg_config, baseline_policy_config)

        # Check if online learning
        self.is_online = getattr(baseline_alg_config, "is_online", None)
        if self.is_online is None:
            self.is_online = False

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

        # Choose initial alg
        self.alg_index = np.random.choice(len(self.alg_set))
        self.set_alg()

    def before_episode(self):
        # Every other trajectory set to exploit
        self.controller.flags.exploit = self.controller.episode_num % 2 == 1

        # Get seed state
        self.traj_seed_state = self.controller.get_seed_state()
    
    def on_update(self):
        # Get seed state
        self.step_seed_state = self.controller.get_seed_state()

    def after_update(self):
        if self.controller.flags.learn:
            if self.is_online:
                self.update_off_policy_learners(self.step_seed_state)
            else:
                self.update_off_policy_learners(self.traj_seed_state)

    def after_episode(self):
        # Update policies after one of each trajectory
        if self.controller.flags.learn and not self.controller.flags.exploit:
            # Select new algorithm
            self.alg_index = self.update_alg()
            print('Selected Index: {}'.format(self.alg_index))

            # Update controller
            self.set_alg()
        
    @abc.abstractmethod
    def update_alg(self):
        raise NotImplementedError()

    def set_alg(self):
        chosen_alg = self.alg_set[self.alg_index]
        self.controller.asys.algorithm = chosen_alg.algorithm
        self.controller.asys.policy_groups[self.pg_id].policy.load(chosen_alg.policy.serialize())
        self.controller.asys.policy_groups[self.pg_id].policy.sampler = chosen_alg.policy.sampler
        self.controller.asys.policy_groups[self.pg_id].model = chosen_alg.model

        self.logger.log_selection(self.controller.episode_num, self.alg_index)

    def update_off_policy_learners(self, seed_state):
        current_seed_state = self.controller.get_seed_state()

        pg = self.get_pg()
        for i in range(self.ensemble_size):
            # Set seed state
            self.controller.set_seed_state(seed_state)

            self.alg_set[i].learn(pg.trajectory)

        # Set seed state
        self.controller.set_seed_state(current_seed_state)
