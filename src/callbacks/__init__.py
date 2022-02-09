"""
Module containing callbacks.

Callbacks define functions that are supplementary in running experiments.
They allow for execution of miscellaneous code at various times in experiments.

The controller will compose the functions of the callbacks together at initialization time,
and call the callbacks in the order that they are defined.
"""

from .base import Callback, CallbackImpl
from .Evaluate import Evaluate
from .LoadPolicy import LoadPolicy
from .PlotReward import PlotReward
from .SaveBest import SaveBest
from .SaveTrajectories import SaveTrajectories
from .Visualize import Visualize
from .Timer import Timer
