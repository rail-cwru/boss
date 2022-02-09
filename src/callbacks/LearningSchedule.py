from typing import TYPE_CHECKING, List

from callbacks import CallbackImpl
from callbacks.base import ScheduleCallback
from config.config import ConfigItemDesc
from . import Callback
from config import gen_check

if TYPE_CHECKING:
    from controller import MDPController
    from config import Config


class LearningSchedule(ScheduleCallback):
    """
    Turn *and keep* Learning on and off at various episodes (before episode).

    Note that special episodes, if they are called directly with controller.run_episode,
        are not subject to this. (Try to not make special episodes that learn, please)
    """

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return [
            ConfigItemDesc(name='schedule',
                           check=gen_check.schedule_of(lambda s: isinstance(s, bool)),
                           info='Dictionary schedule of "on" and "off" at various episode times.')
        ]

    def __init__(self, controller: 'MDPController', config: 'Config'):
        super().__init__(controller, config)
        self.learn_state = False

    def on_trigger(self, value):
        self.learn_state = value

    def before_episode(self):
        super(LearningSchedule, self).before_episode()
        self.controller.flags.learn = self.learn_state
