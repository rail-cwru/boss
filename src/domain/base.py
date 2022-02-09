"""
Domain class.

This acts in a similar manner to the OpenAI "Shape" class, but is modified for specific use in the imeiro framework.

It acts as a NON-WRAPPING accessor for a dtype.

In an ideal world, the accessors would expand via macro into arr[2:5] and other native numpy operations.
In this current non-ideal world, we first create the domain object, then use the index_for_item to extract the slicers.

TODO: Non-wrapping property subject to change (should Domain necessarily contain a np.ndarray?)
"""


import abc
from collections import OrderedDict, defaultdict
from typing import List, Any, Tuple, Union

import numpy as np

from domain.items import DomainItem


class Domain(abc.ABC):

    def __init__(self, items: List['DomainItem']):
        """
        Initialize a domain object.

        It unwinds the shapes with the dtypes of the inputted DomainItems and packs them into a np.ndarray.

        Args:
        TODO - future optimization can determine shortening of bit-depth based on maximum range.
        """
        # Throw an WARNING if you are mixing discrete (int) and continuous (float) dtypes!! We will widen int to float.
        # We might support mixed later, though it kind of messes with our data model.
        widest_dtype = 'int32'
        for item in items:
            dtype = item.dtype

            # Ensure that the dtype is allowed, and then check the range.
            if dtype in ['continuous', 'float', 'float32', 'float64']:
                if dtype == 'float64':
                    widest_dtype = 'float64'
                elif widest_dtype != 'float64':
                    widest_dtype = 'float32'
            elif dtype in ['discrete', 'int', 'int32', 'int64']:
                if 'float' not in widest_dtype and dtype == 'int64':
                    widest_dtype = 'int64'

        # Unwind all shapes into a vector shapes.
        self.__unwound_shapes = OrderedDict([(item.name, np.prod(item.shape)) for item in items])

        # Determine access regions.
        start = 0
        stop = 0
        slicers = OrderedDict()
        for item_name, size in self.__unwound_shapes.items():
            assert size > 0, 'Size cannot be zero but was zero for item [{}]'.format(item_name)
            stop += size
            slicers[item_name] = slice(start, stop)
            start += size
        self.__slicers = slicers

        self.widest_dtype = widest_dtype
        self.discrete = 'int' in self.widest_dtype

        self.shape: Tuple[int, ...] = tuple([sum(self.__unwound_shapes.values())])
        self.items: List[DomainItem] = items

        # TODO change name to something more sane
        # Groups features with the same prefix
        self.prefix_map = defaultdict(list)
        for feature in self.items:
            if feature.prefix:
                self.prefix_map[feature.prefix].append(feature)

    def generate_empty_array(self):
        return np.zeros(self.shape, dtype=self.widest_dtype)

    def item_at_index(self, i: Union[int, List[int]]) -> str:   # maybe Tuple[str, Tuple[int]] ?
        """
        Given the linear index, return the name of the item at that index
        TODO - ...and the position in that item at the index...?

        Call rarely. Should only be for debug / misc info-gathering.
        :param i: linear index
        :return: item at index
        """
        run_index = i
        for name, size in self.__unwound_shapes.items():
            if run_index < size:
                return name
        raise IndexError('Index [{}] exceeded array shape.'.format(i))

    def index_for_name(self, name: str, prefix: Any=None) -> slice:
        """
        Given the item name, return the slice which accesses the data for that item.
        :param item: Item name
        :return: Slice which accesses the item data in array.
        """
        # TODO - It's flattened. Should we reshape a view? Test to see if it passes the view or a copy.
        if prefix is None:
            return self.__slicers[name]
        else:
            return self.__slicers[str(prefix) + '_' + name]

    def get_item_view_by_name(self, data: np.ndarray, name: str, prefix: Any=None) -> np.ndarray:
        """
        Given the item name, return the VIEW into the provided data vector associated with this domain.

        Read/write friendly.

        :param data: The data array that MUST be associated with this domain.
        :param name: Name of feature.
        :param prefix: Prefix of feature (optional)
        :return: View into the data array for the matching item.
        """
        if prefix is None:
            return data[self.__slicers[name]]
        else:
            return data[self.__slicers[str(prefix) + '_' + name]]

    def get_item_view_by_item(self, data: np.ndarray, item: DomainItem):
        """
        Given the item, return the VIEW into the provided data vector associated with this domain.

        Read/write friendly.

        :param data: The data array that MUST be associated with this domain.
        :param item: The item accessed.
        :return: View into the data array for the matching item.
        """
        return data[self.__slicers[item.name]]


    def __eq__(self, other: 'Domain'):
        assert isinstance(other, Domain), "Compared Domain to non-domain object."
        return self.items == other.items

    @classmethod
    def join(cls, *domains):
        """
        Join the domains together. The domains should all be of the same type.

        This calls the __default_join method of the domain's class.
        Some domains in the domain type hierarchy may implement this differently if their
            __init__ signature is different than the [items] signature.
        :param domains:
        :return:
        """
        raise NotImplementedError()


