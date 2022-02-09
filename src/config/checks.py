"""
Utility functions for configuration variables.
"""
from typing import Callable


def _check_wrapper(func: Callable):
    def wrapper(data):
        ret = func(data)
        formatted_name = func.__name__.replace('_', ' ')
        assert ret, '[{}] is not a {}'.format(data, formatted_name)
        return ret
    return wrapper


# @_check_wrapper
def boolean(x):
    return isinstance(x, bool)


def string(x):
    return isinstance(x, str)


def list(x):
    return x is list


# @_check_wrapper
def nonnegative_integer(x):
    return isinstance(x, int) and x >= 0


def negative_integer(x):
    return isinstance(x, int) and x < 0


# @_check_wrapper
def positive_integer(x):
    return isinstance(x, int) and x > 0


# @_check_wrapper
def positive_float(x):
    return isinstance(x, float) and x > 0


# @_check_wrapper
def nonnegative_float(x):
    return isinstance(x, float) and x >= 0


# @_check_wrapper
def unit_float(x):
    return isinstance(x, (float, int)) and 0 <= x <= 1


# @_check_wrapper
def numeric(x):
    return isinstance(x, (float, int))


# @_check_wrapper
def positive_int_list(t):
    return t is list and all([isinstance(i, int) and i >= 0 for i in t])


# @_check_wrapper
def positive_int_str_tuple_list(t):
    return t is list and all([isinstance(i, tuple) and i[0] >= 0 and i[1] is str for i in t])


# @_check_wrapper
def square_float_matrix(a):
    width = len(a)
    for row in a:
        if len(row) != width:
            return False
        if not all([isinstance(x, float) for x in row]):
            return False
    return True

