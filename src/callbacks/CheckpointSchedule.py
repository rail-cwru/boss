from typing import TYPE_CHECKING, List

import copy

from callbacks.base import ScheduleCallback
from common.domain_transfer import DomainTransferMessage
from config.config import ConfigItemDesc
from config import gen_check

if TYPE_CHECKING:
    from controller import MDPController
    from config import Config


class CheckpointSchedule(ScheduleCallback):
    """
    Save and load the status of the controller (env + asys) before various episodes.
    """

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return [
            ConfigItemDesc(name='schedule',
                           # Can only check at init time...
                           check=gen_check.schedule_of(lambda s: s in ['save', 'load']),
                           info='Dictionary schedule of "save" or "load" at various episode times.')
        ]

    def __init__(self, controller: 'MDPController', config: 'Config'):
        super().__init__(controller, config)
        assert self.schedule[sorted(self.schedule.keys())[0]] == 'save', 'Cannot load without saving first.'
        self.checkpoint = None

    def on_trigger(self, value):
        if value == 'save':
            self.checkpoint = self.controller.get_checkpoint()
        elif value == 'load':
            self.controller.set_from_checkpoint(self.checkpoint)

