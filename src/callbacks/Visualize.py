from typing import TYPE_CHECKING, List

from callbacks.base import CallbackImpl
from config.config import ConfigItemDesc

from config import checks
from . import Callback

if TYPE_CHECKING:
    from config import Config
    from controller import MDPController


class Visualize(Callback):
    """Environment visualizer.

    Basic visualizations. This will be able to visualize different types of
    environments and how all the agents within that environment move.
    """

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return [
            ConfigItemDesc(name='timestep',
                           check=checks.positive_integer,
                           info='Interval of episodes after which to run this callback.'),
        ]

    def __init__(self, controller: 'MDPController', config: 'Config'):
        super().__init__(controller, config)
        # Visualize every [timestep] episodes.
        self.timestep = config.callbacks['Visualize'].timestep

    def _get_implement_flags(self):
        return CallbackImpl(on_episode_start=True, after_episode=True)

    def on_episode_start(self):
        if self.controller.episode_num % self.timestep == 0:
            self.controller.flags.visualize = True

    def after_episode(self):
        self.controller.flags.visualize = False
