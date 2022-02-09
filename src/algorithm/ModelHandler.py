from typing import Dict, Callable

from config.moduleframe import AbstractModuleFrame
from model import Model
from policy import Policy
from common.PropertyClass import PropertyClass

class ModelHandler(PropertyClass):
    """
    Create new ModelHandler (concrete)

    This class is used to support multiple inheritance chain.
    Every algorithm class should implement ModelHandler either directly or through Algorithm
    Even though this class is concrete, in practice it will always act as a parent.
    """

    @classmethod
    def end_episode(cls, policy: Policy, model: Model):
        """
        This method implements any cleanup/algorithmic procedure algorithm needs to do to end the episode
        This method uses functional inheritance and should always call super().end_episode() in an
        implementing class
        :return:  None
        """
        cls.end_episode_helper(policy, model)

        if cls != ModelHandler:
            for parent in cls.__bases__:
                if parent != AbstractModuleFrame:
                    parent.end_episode(policy, model)

    @classmethod
    def end_episode_helper(cls, policy, model):
        """
        Default end episode functionality. Doesn't do anything.
        This implements whatever the class personally (in terms of the hierarchy) needs to do at the end of the episode
        :param policy:
        :param model:
        :return: None
        """
        pass

    @classmethod
    def make_model(cls, policy, config):
        """
        Returns a Model class containing whatever ElementModels is needs to function
        This method is uses functional inheritance and should always do
        # model = Model()
        # ...
        # return model.merge_model(super().make_model())
        in an implementing class
        :return: Model
        """
        # Current class adds what it needs the model
        model = cls.make_model_helper(policy, config)

        # Each parent add what they need to the model
        if cls != ModelHandler:
            for parent in cls.__bases__:
                if getattr(parent, 'make_model', None):
                    model.merge_model(parent.make_model(policy, config))

        return model

    @classmethod
    def make_model_helper(cls, policy, config):
        """
        Default helper functionality. Just makes an empty model.
        Defines whatever ElementModel(s) an algorithm personally (in terms of the hierarchy) needs
        :param policy:
        :param config:
        :return:
        """
        return Model()


