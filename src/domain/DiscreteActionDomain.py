import numpy as np

from domain.ActionDomain import ActionDomain
from domain.actions import Action, DiscreteAction, ActionType


class DiscreteActionDomain(ActionDomain):
    """
    A special subclass of ActionDomain which functionally has one single discrete DomainItem,
        from which one action can be chosen.

    If instantiated with a list of discrete DomainItems of ranges=[r0, r1, ...] it will produce
        a single DomainItem with a range of np.prod([ranges]).

    This features a method to extract the sub-actions from a compound action.

    Some environments support only this sort of action.
    """
    # If the ActionDomain only contains one discrete action which corresponds to the range of exclusive actions

    def __init__(self, items: [DiscreteAction], num_agents: int):
        self.ranges = np.zeros(len(items), dtype='int32')
        self.strides = np.zeros(len(items), dtype='int32')
        # np.random.seed(12345)
        if len(items) > 1:
            full_range = 1
            # The compound item's index will correspond thus:
            # i0 + r0*i1 + r0*r1*i2 + ...
            for reverse_i, item in enumerate(reversed(items)):
                assert item.feature_type == ActionType.DISCRETE,\
                    'The DomainItems in a DiscreteIndexedActionDomain must be discrete. Instead, got [{}]'\
                    .format(item)
                i = len(items) - reverse_i - 1
                item_size = item.range.stop - item.range.start
                self.strides[i] = full_range
                full_range *= item_size
                self.ranges[i] = item_size
                assert item.range.start == 0, 'Discrete DomainItem for Actions must have range starting with 0.'
            item: Action = DiscreteAction('compound_item', full_range)
            self.sub_actions = items
            self.order = len(items)
            self.is_compound = True
            self.full_range = full_range
        else:
            item = items[0]
            self.ranges[0] = item.range.stop - item.range.start
            self.sub_actions = [item]
            self.order = 1
            self.is_compound = False
            self.full_range = self.ranges[0]
        super().__init__([item], num_agents=num_agents)
        self.discrete_exclusive = True

    def random_sample(self):
        return np.random.randint(self.full_range)

    def extract_sub_actions(self, data: int):
        """
        Extract the compound action into the sub-actions, if applicable.
        :param data:
        :return:
        """
        assert self.is_compound, 'The DiscreteIndexedActionDomain is not compound and cannot extract sub-actions.'
        # Faster than reusing arrays. Do not change.
        joint = np.remainder(np.floor_divide(data, self.strides), self.ranges)
        # This is faster than np.split. Do not change.
        return np.array(joint)

    def make_compound_action(self, *data):
        """
        Transform the sub actions into the compound action index.
        Make sure that the compound action domain is being passed action data which corresponds to its sub-actions,
            in the order that they were used to construct the original ActionDomain.
        :param data: Sequence of np.ndarrays which contain action data corresponding to the sub-actions of the
                      DiscreteIndexedActionDomain.
        :return: Single data which is the index of the compound discrete action.
        """
        assert self.is_compound, 'The DiscreteIndexedActionDomain is not compound and cannot extract sub-actions.'
        return np.dot(self.strides, np.hstack(data)).reshape(1)
