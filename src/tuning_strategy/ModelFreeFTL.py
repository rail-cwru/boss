import os
from typing import List

from config.config import ConfigItemDesc
from .FTL import FTL
from .utils import UCB, OpeMemory, IS, PF, ope_PF, ope_IS


class ModelFreeFTL(FTL):
    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return FTL.get_class_config() + [
            ConfigItemDesc(name='off_policy_estimator',
                           check=lambda s: isinstance(s, str) and (s in [ope_PF, ope_IS]),
                           info='OPE to use when choosing an algorithm.'),
            ConfigItemDesc(name='ope_parameters',
                           check=lambda s: isinstance(s, dict),
                           info='Required parameters for OPE function.'),
            ConfigItemDesc(name='ope_memory',
                           check=lambda s: isinstance(s, int),
                           info='Memory size for OPE calculations.'),
            ConfigItemDesc(name='ucb_alpha',
                           check=lambda s: isinstance(s, float),
                           info='Alpha value to use in UCB (std dev).')            
        ]

    def __init__(self, controller: 'MDPController', config: 'Config', pg_id: int):
        super().__init__(controller, config, pg_id)

        self.ucb_calc = UCB(self.config.ucb_alpha, self.config.ensemble_size)
        self.ope_memory = OpeMemory(self.config.ope_memory)

        # Specify off policy estimator functions
        ope_params = self.config.ope_parameters
        if self.config.off_policy_estimator == ope_IS:
            self.ope_params = [ope_params['is_weighted']]
            self.ope_func = IS
        elif self.config.off_policy_estimator == ope_PF:
            self.ope_params = [ope_params['resample_count']]
            self.ope_func = PF

    def before_run(self):
        # Reset ucb and memory
        self.ucb_calc.reset()
        self.ope_memory.reset()

        super().before_run()

    def after_episode(self):
        if self.controller.flags.learn and self.controller.flags.exploit:
            # Update OPE memory with exploit learn trajectories
            pg = self.get_pg()
            self.ope_memory.log(pg.trajectory)

        super().after_episode()

    def update_alg(self):
        for j_alg in range(self.ensemble_size):
            # Find probabilities of actions by evaluation policy
            D = self.ope_memory.get_ope_trajectories(self.alg_set[self.alg_index], self.alg_set[j_alg], self.pg_id)

            # Calculate estimated average cumulative reward
            self.ucb_calc.log(j_alg, *self.ope_func(D, self.ope_params))

        # Choose best algorithm based on UCB, after every arm has been pulled once
        return self.ucb_calc.calculate(self.result_file, self.alg_index)