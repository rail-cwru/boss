import abc
from typing import List

from common.properties import Properties
from config.moduleframe import AbstractModuleFrame
from config.config import ConfigItemDesc


class PropertyClass(AbstractModuleFrame):

    @classmethod
    @abc.abstractmethod
    def get_class_config(cls) -> List['ConfigItemDesc']:
        raise NotImplementedError()

    @classmethod
    def properties(cls) -> Properties:
        properties = cls.properties_helper()
        if cls != PropertyClass:
            for parent in cls.__bases__:
                if parent != AbstractModuleFrame:
                    properties.merge(parent.properties())

        return properties

    @classmethod
    def properties_helper(cls) -> Properties:
        return Properties()
