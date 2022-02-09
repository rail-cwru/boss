import pickle
from callbacks import CallbackImpl
from config.config import ConfigItemDesc
from . import Callback

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from config import Config
    from controller import MDPController


class SaveTrajectories(Callback):
    """
    Save the learning trajectories to a file at the end of episode.
    """

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return [
            ConfigItemDesc(name='file_location',
                           check=lambda s: isinstance(s, str),
                           info='File to save trajectories to'),
        ]

    def __init__(self, controller: 'MDPController', config: 'Config'):
        super().__init__(controller, config)
        # To save to
        self.file_location = config.callbacks['SaveTrajectories'].file_location

    def _get_implement_flags(self):
        return CallbackImpl(finalize=True)

    def finalize(self):
        with open(self.file_location, mode='wb+') as f:
            pickle.dump(self.controller.episode_trajectories, f)

