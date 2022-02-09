from . import Callback
from tuning_strategy import TuningStrategy
from callbacks import CallbackImpl
from config.config import ConfigItemDesc, ConfigDesc

import os
import numpy as np
from typing import TYPE_CHECKING, List, Dict, Any

if TYPE_CHECKING:
    from config import Config
    from controller import MDPController


class TuneHyperparameters(Callback):
    """
    Tune hyperparameters and save the tuning results to a file at the end of the experiment.
    """

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return [
            ConfigDesc('tuning_strategy', 'Tuning strategy config', default_configs=[])
        ]

    def __init__(self, controller: 'MDPController', config: 'Config'):
        super().__init__(controller, config)
        self.tuners = {}
        tuning_class = config.callbacks['TuneHyperparameters'].tuning_strategy.module_class
        for pg in self.controller.asys.policy_groups:
            self.tuners[pg.pg_id] = tuning_class(controller, config, pg.pg_id)
        
    def _get_implement_flags(self):
        return CallbackImpl(before_run=True, before_episode=True, on_update=True, after_update=True, after_episode=True, after_run=True, finalize=True)

    def before_run(self):
        for _, tuner in self.tuners.items():
            tuner.before_run()

    def before_episode(self):
        for _, tuner in self.tuners.items():
            tuner.before_episode()

    def on_update(self,
                  agents_observation: Dict[Any, np.ndarray],
                  agent_action_map: Dict[Any, np.ndarray],
                  agent_rewards: Dict[Any, float]):
        for _, tuner in self.tuners.items():
            tuner.on_update()
        return agents_observation, agent_action_map, agent_rewards

    def after_update(self):
        for _, tuner in self.tuners.items():
            tuner.after_update()

    def after_episode(self):
        for _, tuner in self.tuners.items():
            tuner.after_episode()

    def after_run(self):
        for _, tuner in self.tuners.items():
            tuner.after_run()

    def finalize(self):
        for _, tuner in self.tuners.items():
            tuner.finalize()

    