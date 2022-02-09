"""
Arbitrarily defined functions that can be called from within predicates
Newly added functions must be added to the "dispatcher" dictionary at the end of the file
"""

from typing import List


def manhattan_distance(loc1: List[int], loc2: List[int], direction=None) -> int:
    """
    Calculates the manhattan distance between two locations
    :param loc1: The base location
    :param loc2: The second location to be compared to the base
    :param direction: An optional parameter restricting the calculation to one direction
    :return: the manhattan distance
    """
    y_dist = loc2[0] - loc1[0]
    x_dist = loc2[1] - loc1[1]

    if direction is None:
        return abs(y_dist) + abs(x_dist)
    elif direction == 0:  # North
        return -y_dist
    elif direction == 1:  # East
        return x_dist
    elif direction == 2:  # South
        return y_dist
    elif direction == 3:  # West
        return -x_dist
    else:
        raise ValueError('Invalid direction passed to euclidean_distance in predicate')


dispatcher = {"manhattan_distance": manhattan_distance}
