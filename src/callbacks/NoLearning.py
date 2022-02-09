from callbacks import CallbackImpl
from . import Callback


class NoLearning(Callback):
    """
    Prevents learning from being used in the run.
    """

    def _get_implement_flags(self):
        return CallbackImpl(before_episode=True)

    def before_episode(self):
        self.controller.flags.learn = False
