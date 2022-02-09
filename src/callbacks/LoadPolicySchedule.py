from typing import TYPE_CHECKING, List

import os

from callbacks.base import ScheduleCallback
from config.config import ConfigItemDesc
from config import gen_check

if TYPE_CHECKING:
    from controller import MDPController
    from config import Config


class LoadPolicySchedule(ScheduleCallback):
    """
    Load an agentsystem's learned policies before certain episodes.

    Loaded policy must be compatible.
    """

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return [
            ConfigItemDesc(name='schedule',
                           check=gen_check.schedule_of(lambda s: isinstance(s, str)),
                           info='Dictionary schedule of saved policy npz to load at various episode times.')
        ]

    def __init__(self, controller: 'MDPController', config: 'Config'):
        super().__init__(controller, config)
        for v in self.schedule.values():
            assert os.path.exists(v), 'The file [{}] to load policy from does not exist.'.format(v)

    def on_trigger(self, value):
        self.controller.asys.load(self.schedule[self.controller.episode_num])
