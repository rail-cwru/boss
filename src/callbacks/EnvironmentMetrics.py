"""
Statistics module.
"""

from typing import TYPE_CHECKING, List
from callbacks import Callback, CallbackImpl
from matplotlib import pyplot as plt
import numpy as np

from config.config import ConfigItemDesc

if TYPE_CHECKING:
    from config import Config
    from controller import MDPController


class EnvironmentMetrics(Callback):
    """
    Get some sort of metric from the environment after each episode.

    Returns data into "finalize."
    """

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return []

    def __init__(self, controller: 'MDPController', config: 'Config'):
        super().__init__(controller, config)
        self.metrics = {}   # episode num -> metrics

    def _get_implement_flags(self):
        return CallbackImpl(before_run=True, after_episode=True, finalize=True)

    def before_run(self):
        self.controller.env.set_gather_metrics(True)

    def after_episode(self):
        self.metrics[self.controller.episode_num] = self.controller.env.get_metrics()

    def finalize(self):
        return {'metrics': self.metrics}
