from typing import Callable, Union
from pathlib import Path


Schedule = Callable[[float], float]



def constant_fn(val: float) -> Schedule:
    """
    Create a function that returns a constant
    It is useful for learning rate schedule (to avoid code duplication)

    :param val:
    :return:
    """

    def func(_):
        return val

    return func


def get_schedule_fn(value_schedule: Union[Schedule, float, int]) -> Schedule:
    """
    Transform (if needed) learning rate and clip range (for PPO)
    to callable.

    :param value_schedule:
    :return:
    """
    # If the passed schedule is a float
    # create a constant function
    if isinstance(value_schedule, (float, int)):
        # Cast to float to avoid errors
        value_schedule = constant_fn(float(value_schedule))
    else:
        assert callable(value_schedule)
    return value_schedule


