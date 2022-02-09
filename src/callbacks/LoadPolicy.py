from typing import TYPE_CHECKING, List

from callbacks import CallbackImpl
from config.config import ConfigItemDesc
from . import Callback
if TYPE_CHECKING:
    from controller import MDPController
    from config import Config


class LoadPolicy(Callback):
    """
    Load an agentsystem's learned policies at the beginning of the run.

    Must be compatible.
    """

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return [
            ConfigItemDesc(name='file_location',
                           check=lambda s: isinstance(s, str),
                           info='Location of policy npz file to load')
        ]

    def __init__(self, controller: 'MDPController', config: 'Config'):
        super().__init__(controller, config)
        callback_config = config.callbacks['LoadPolicy']
        # Location to load from
        self.file_location = callback_config.file_location

    def _get_implement_flags(self):
        return CallbackImpl(before_run=True)

    def before_run(self):
        self.controller.asys.load(self.file_location)
