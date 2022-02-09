import abc
import warnings
from typing import List, TYPE_CHECKING
import os

from common.properties import Properties
from config.moduleframe import AbstractModuleFrame
from config.config import ConfigItemDesc

if TYPE_CHECKING:
    from controller import MDPController

class TuningStrategy(AbstractModuleFrame):
    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return [
            ConfigItemDesc(name='file_location',
                           check=lambda s: isinstance(s, str),
                           info='File to save parameter tracking results to.'),
            ConfigItemDesc(name='tune_policy_sampler_params',
                           check=lambda s: isinstance(s, bool),
                           info='Indicator whether or not to tune the policy sampler parameters in addition to the algorithm parameters.')
        ]

    @classmethod
    def properties(cls):
        return Properties()
    
    def __init__(self, controller: 'MDPController', config: 'Config', pg_id: int):
        self.config = config.callbacks['TuneHyperparameters'].tuning_strategy
        self.pg_id = pg_id
        self.controller = controller
        self.tune_policy_sampler_params = self.config.tune_policy_sampler_params

        file_name = '{}_{}_{}_{}'.format(self.config.name, pg_id, config.algorithm.name, config.environment.name)
        self.result_file = os.path.join(self.config.file_location, file_name)

        warnings.warn('Tuning Strategies have only been tested on independent system for discrete actions.')

    def get_pg(self):
        return self.controller.asys.policy_groups[self.pg_id]

    def before_run(self):
        pass

    def before_episode(self):
        pass

    def on_update(self):
        pass

    def after_update(self):
        pass

    def after_episode(self):
        pass
    
    def after_run(self):
        pass

    def finalize(self):
        pass