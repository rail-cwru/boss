from copy import deepcopy

from typing import List

import numpy as np

from domain.actions import Action
from domain.base import Domain

class ActionDomain(Domain):
    """
    A Domain which describes possible Actions which can be done in the environment,
    """
    def __init__(self, items: List[Action], num_agents: int=None):
        super(ActionDomain, self).__init__(items)
        self.num_agents = num_agents  # TODO do away with num_agents in favor of some better data serialization strategy
        self.items: List[Action] = self.items
        self.is_compound = False

    def random_sample(self):
        """
        Sample a random action from the domain.
        The domain must not contain infinite-range values.
        :return: A random action from the domain.
        """
        arr = np.random.rand(*self.shape)
        for item in self.items:
            slicer = self.index_for_name(item.name)
            # TODO determine behavior of random sampling on open domains.
            assert item.range.start is not None, 'INFINITE RANGE SAMPLING NOT SUPPORTED'
            assert item.range.stop is not None, 'INFINITE RANGE SAMPLING NOT SUPPORTED'
            diff = item.range.stop - item.range.start
            np.subtract(arr[slicer], item.range.start, out=arr[slicer])
            if item.discrete:
                np.floor_divide(arr[slicer], diff, out=arr[slicer])
            else:
                np.floor_divide(arr[slicer], diff, out=arr[slicer])
        return arr.astype(self.widest_dtype)

    def get_action_range(self):
        """
        Get a python range of possible values for the action to take on.
        This only supports single-valued actions.
        :return: Python range of possible action values.
        """
        action_range = []

        for action_item in self.items:
            a_slice = action_item.range
            a_range = range(a_slice.start, a_slice.stop)
            action_range.append(a_range)

        return action_range

    @classmethod
    def join(cls, *domains):
        all_items = []
        domain_cls = type(domains[0])
        for i, domain in enumerate(domains):
            assert type(domain) == domain_cls, 'Cannot join Domains of different class, even inherited ones.'
            all_items += [item.copy_with_added_prefix(str(i)) for item in domain.items]
        return domain_cls(all_items, None)
