from typing import List, TYPE_CHECKING
import os
import abc

from config import ConfigItemDesc, checks
from .base import TuningStrategy
from .utils import OpeMemory, IS, KL

if TYPE_CHECKING:
    from config import Config
    from controller import MDPController

class HOOF(TuningStrategy):
    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return TuningStrategy.get_class_config() + [
            ConfigItemDesc(name='ensemble_size',
                           check=lambda s: isinstance(s, int),
                           info='Number of parameter configurations.'),
            ConfigItemDesc(name='max_kl',
                           check=checks.positive_float,
                           info='Maximum KL distance.')
        ]

    def __init__(self, controller: 'MDPController', config: 'Config', pg_id: int):
        super().__init__(controller, config, pg_id)
        self.ensemble_size = self.config.ensemble_size
        self.max_kl = self.config.max_kl

    def update_alg(self):
        # Stash current policy, model and seed
        current_policy = self.controller.asys.policy_groups[self.pg_id].policy.serialize()
        current_model = self.controller.asys.policy_groups[self.pg_id].model
        traj_seed_state = self.controller.get_seed_state()

        # Draw new parameters
        # TODO

        results = {}
        for j in range(self.ensemble_size):
            # Reset policy, model and seed
            self.controller.asys.policy_groups[self.pg_id].policy.load(current_policy)
            self.controller.asys.policy_groups[self.pg_id].model = current_model
            self.controller.set_seed_state(traj_seed_state)

            # Adjust parameters
            # TODO

            # Run episode
            self.controller.run_episode()

            # Compute expected reward
            D = []
            reward, _ = IS(D, [True])

            # Compute KL distance
            kl_dist = KL(new_mean, new_sd, old_mean, old_sd)

            results[j] = (reward, kl_dist)

        # Choose best algorithm based on peeking ahead
        max_index = []
        max_reward = sys.float_info.min
        for index, reward, kl_dist in results.items():
            if kl_dist < self.max_kl:
                if reward > max_reward:
                    max_index = [index]
                elif reward == max_rward:
                    max_index.append(index)

        # Random tiebreak
        chosen_index = np.random.choice(max_index)

        # Update parameters on controller
        # TODO

        # Reset policy, model and seed
        self.controller.asys.policy_groups[self.pg_id].policy.load(current_policy)
        self.controller.asys.policy_groups[self.pg_id].model = current_model
        self.controller.set_seed_state(traj_seed_state)

    def after_episode(self):
        if self.controller.flags.learn:
            self.update_alg()
