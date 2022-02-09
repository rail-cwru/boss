import abc
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from common.properties import Properties
    from config.config import ConfigItemDesc


class AbstractModuleFrame(abc.ABC):
    """
    A interface-like abstract class for classes which must implement get_class_config() and properties()
    """

    @classmethod
    @abc.abstractmethod
    def get_class_config(cls) -> List['ConfigItemDesc']:
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def properties(cls) -> 'Properties':
        raise NotImplementedError()
