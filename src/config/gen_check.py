"""
General checks.

Check functions which are more general and should be used with the inbuilt functools partial and/or functions from
    "checks.py"
"""
from config import checks


def n_list_func(n, func, data):
    """
    List is length n and all elements fulfill [func]
    """
    return isinstance(data, list) and len(data) == n and all([func(x) for x in data])


def nonempty_list_func(func, data):
    """
    List is nonempty and all elements fulfill [func]
    """
    return isinstance(data, list) and len(data) > 0 and all([func(x) for x in data])


def uniform_dict(names, func, data):
    """
    Check data is a dictionary having keys from [names] have data fulfilling [func]
    """
    return isinstance(data, dict) and all([name in data and func(data[name]) for name in names])


def schedule_of(value_check):
    err_msg = 'A schedule must be a dictionary mapping episode number to values.'

    def _valid_schedule(schedule):
        assert isinstance(schedule, dict), err_msg
        for k, v in schedule.items():
            assert checks.positive_integer(int(k)), err_msg
            assert value_check(v), 'The value [{}] was not a valid schedule item.'.format(v)
        return True

    return _valid_schedule
