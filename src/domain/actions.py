from typing import List, Any, Optional

from domain.items import DomainItem
from enum import Enum


class ActionType(Enum):
    NONE = 0
    DISCRETE = 1  # An action describing a choice out of discrete set of actions.
    REAL = 2
    # INT_VECTOR = 3 # An action described by a product of discrete spaces. e.g. a move on a discrete coordinate grid.
    # REAL_VECTOR = 4 # An action described by a real-valued vector


class Action(DomainItem):
    """
    DomainItems used for observation / environment state representation.
    """

    def __init__(self, name: str,
                 shape: List[int],
                 dtype: str,
                 drange: slice=slice(None),
                 prefix: Any=None,
                 feature_type=ActionType.NONE):
        """
        A typeless action.

        Use sparingly.

        :param name: Name of action
        :param shape: Shape taken on by the data which this action is written to
        :param dtype: The datatype that should represent the action
        :param drange: The range of values that the action can take on, represented as a slice
        :param prefix: A prefix for the name of this action
        :param feature_type: The ActionType of this action
        """
        self.feature_type = feature_type
        super(Action, self).__init__(name, shape, dtype, drange, prefix)


class DiscreteAction(Action):

    def __init__(self, name: str, num_actions: int, prefix: Any=None):
        """
        A action representing a selection from a discrete set of actions.

        :param name: Name of this action
        :param num_actions: The number of possible actions described by this Action
        :param prefix: Prefix to name (optional)
        """
        super().__init__(name=name,
                         shape=[1],
                         dtype='int64',
                         drange=slice(0, num_actions),
                         prefix=prefix,
                         feature_type=ActionType.DISCRETE)


class RealAction(Action):

    def __init__(self, name: str, min: Optional[float]=None, max: Optional[float]=None, prefix: Any=None):
        """
        A action representing a real value.

        :param name: Name of this action
        :param min: The minimum value this action can take on.
        :param max: The maximum value this action can take on.
        :param prefix: Prefix to name (optional)
        """

        super().__init__(name=name,
                         shape=[1],
                         dtype='float64',
                         drange=slice(min, max),
                         prefix=prefix,
                         feature_type=ActionType.REAL)
