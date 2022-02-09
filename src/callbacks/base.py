"""
Generic callback class. 

Any function not crucial or central to the system which may intercept data 
passed between objects in the controller.


"""

import abc
from typing import Dict, Any, TYPE_CHECKING, List

import numpy as np

from common.properties import Properties
from config.moduleframe import AbstractModuleFrame
from config.config import ConfigItemDesc

if TYPE_CHECKING:
    from config import Config
    from controller import MDPController


class CallbackImpl(object):
    """
    Class which describes which functions Callback implements.
    """

    keys = ['before_run',
            'before_episode',
            'on_episode_start',
            'before_observe',
            'on_observe',
            'on_action',
            'on_update',
            'after_update',
            'after_episode',
            'after_run',
            'finalize']

    def __init__(self,
                 before_run=False,
                 before_episode=False,
                 on_episode_start=False,
                 before_observe=False,
                 on_observe=False,
                 on_action=False,
                 on_update=False,
                 after_update=False,
                 after_episode=False,
                 after_run=False,
                 finalize=False):
        self.flags = {
            'before_run': before_run,
            'before_episode': before_episode,
            'on_episode_start': on_episode_start,
            'before_observe': before_observe,
            'on_observe': on_observe,
            'on_action': on_action,
            'on_update': on_update,
            'after_update': after_update,
            'after_episode': after_episode,
            'after_run': after_run,
            'finalize': finalize
        }


class Callback(AbstractModuleFrame):
    """
    Generic callback class.

    Defines functions that are critical to the operation of the system. These 
    functions are designed to intercept data passed between objects in the controller.

    The controller will compose the functions of the callbacks together to alter its
    own intercepting functions at initialization time.

    Params:
    -------
    controller: MDPController
        Specific controller for different RL environments.

    config: dict[str, Any]
        Contains specific properties used for this experiment.
    """

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return []

    @classmethod
    def properties(cls):
        return Properties()

    def __init__(self, controller: 'MDPController', config: 'Config'):
        """Initialize callback."""
        self.controller = controller
        self.__implement_flags = self._get_implement_flags()

    def implements(self, func_name):
        # TODO (future) untrustworthy; not idiotproof. automagic determination eventually, please
        return self.__implement_flags.flags[func_name]

    @abc.abstractmethod
    def _get_implement_flags(self) -> CallbackImpl:
        """
        Obtain a dictionary containing the implementation of each function
        in the callback class so that Controller can avoid calling empty functions.
        :return: A dictionary
        """
        raise NotImplementedError()

    def before_run(self):
        """
        Called before the main run.
        """
        pass

    def before_episode(self):
        """
        Called before the episode is run via MDPController.run_episode.
        """

    def on_episode_start(self):
        """
        Called after the environment is reset at the beginning of an episode, inside run_episode.
        """
        pass

    def before_observe(self):
        """
        Called before observe at each step during an episode.
        """
        pass

    def on_observe(self, agents_observation: Dict[Any, np.ndarray]) -> Dict[Any, np.ndarray]:
        """
        Intercepts the observation from the Environment to the Agentsystem during an episode.
        """
        pass

    def on_action(self,
                  agents_observation: Dict[Any, np.ndarray],
                  agent_action_map: Dict[Any, np.ndarray]) -> (Dict[Any, np.ndarray], Dict[Any, np.ndarray]):
        """
        Intercepts the actions passed from the Agent System to the Environment.
        """
        pass

    def on_update(self,
                  agents_observation: Dict[Any, np.ndarray],
                  agent_action_map: Dict[Any, np.ndarray],
                  agent_rewards: Dict[Any, float]) -> (Dict[Any, float], Dict[Any, np.ndarray], Dict[Any, np.ndarray]):
        """
        Intercepts the learn update step.
        """
        pass

    def after_update(self):
        """
        Called at the end of an episode step.
        """
        pass

    def after_episode(self):
        """
        Called after an episode, after the trajectory has been added to the controller's record.
        """
        pass

    def after_run(self):
        """
        Called after all episodes have terminated.
        """
        pass

    def finalize(self):
        """
        Called after after_run after the episodes have terminated.
        """
        pass


class ScheduleCallback(Callback):

    def __init__(self, controller: 'MDPController', config: 'Config'):
        super().__init__(controller, config)
        self.callback_config = config.callbacks[self.__class__.__name__]
        self.schedule = {int(episode): value for episode, value in self.callback_config.schedule.items()}

    def _get_implement_flags(self) -> CallbackImpl:
        return CallbackImpl(before_episode=True)

    def before_episode(self):
        if self.controller.episode_num in self.schedule:
            self.on_trigger(self.schedule[self.controller.episode_num])

    @abc.abstractmethod
    def on_trigger(self, value):
        """
        Called with the value of the content of the schedule at the schedule time.
        """
        raise NotImplementedError()


