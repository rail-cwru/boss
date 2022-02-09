import abc
from typing import Dict, Callable, List, ClassVar

from config import Config
from common.properties import Properties


class Model(abc.ABC):
    def __init__(self, element_models=None):
        """
        Creates a new Model (concrete)

        This creates a new Model object whose attributes are dynamically determined
        Attributes will be other classes that represent more specific models (ElementModels)
        :param element_models: Class or list of classes (not instances) to have as attributes
        """
        self.attr_list = []
        if isinstance(element_models, ElementModel):
            element_model = element_models
            model_name = element_model.get_name()
            setattr(self, model_name, element_model)
            self.attr_list.append(model_name)

        elif type(element_models) == list:
            for element_model in element_models:
                model_name = element_model.get_name()
                setattr(self, model_name, element_model)
                self.attr_list.append(model_name)

    def merge_model(self, model: 'Model'):
        """
        Merge extra components into model
        :param model: Model to merge
        :return:
        """
        attrs = set(model.attr_list)
        disjunction = [x for x in attrs if x not in self.attr_list]

        for attr in disjunction:
            setattr(self, attr, getattr(model, attr))


class ElementModel(abc.ABC):

    @classmethod
    @property
    def config_signature(cls) -> Dict[str, Callable]:
        sig = {
            # Any configurable fields REQUIRED OF ALL Models should go here.
        }
        return sig

    @classmethod
    @property
    @abc.abstractmethod
    def properties(cls) -> Properties:
        """
        Return the compatibility properties of the class.

        This must be implemented
        """
        return Properties()

    def __init__(self, config: Config):
        """
        Instantiate a new ElementModel (abstract)
        
        Base class for ElementModel used by Model.
        This model keeps track of just one element of the world.
        Compose ElementModels with Model to form complete model
        Models vary greatly so this base class serves mostly as a placeholder
        """
        self.config = config

    @staticmethod
    @abc.abstractmethod
    def get_name():
        """
        Returns the name this class will be accessed with as an attribute of Model
        :return: name
        """
        raise NotImplementedError
