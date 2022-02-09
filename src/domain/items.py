from copy import deepcopy

from typing import List, Any, T

import numpy as np


class DomainItem(object):
    """
    A single atom of meaningful data that could be contained in a larger domain.
    """

    def __init__(self, name: str,
                 shape: List[int],
                 dtype: str,
                 drange: slice=slice(None),
                 prefix: Any=None):
        """
        Initialize a domain item.

        Use sparingly.

        items: List[str]
            The names of the information represented by the data array.
        shapes: Dict[str, List[int]]
            The shapes of the data.
        dtypes: Dict[str, Union[np.dtype, str]]
            We will treat continuous and discrete respectively as 'float32' and 'int32'
            Unless float64 and int64 are explicitly specified.
        ranges: slice
            The valid range of values the data this item represents can take on.
            A slice for a discrete item describes the non-inclusive integer range spanned by the slice.
                e.g. slice(0, 5) => [0, 1, 2, 3, 4]
            A slice for a continuous item describes the upper-open real range spanned by the slice.
                e.g. slice(0, 5) => [0, 5)
        prefix: Any
            A str-able prefix to prepend to the ID. e.g. agent IDs for agent-specific domain items.
        """
        self.raw_name = name
        if prefix is not None:
            name = str(prefix) + '_' + name
        # TODO prevent name prefix infinite compounding via discrete indexed action domain
        self.name = name
        self.shape = shape
        self.range = drange
        self.prefix = prefix

        bad_range_msg = 'The dtype [{}] does not accommodate the range for item {}'.format(dtype, name)

        dtype = dtype.lower()
        if dtype in ['continuous', 'float', 'float32', 'float64']:
            np_dtype = dtype if dtype != 'continuous' else 'float32'
            info = np.finfo(np_dtype)
            self.discrete = False
        elif dtype in ['discrete', 'int', 'int32', 'int64']:
            np_dtype = dtype if dtype != 'discrete' else 'int32'
            info = np.iinfo(np_dtype)
            self.discrete = True
        else:
            raise ValueError('Unrecognized dtype [{}] for item [{}]'.format(dtype, name))

        bad_hi = drange.stop is not None and info.max < drange.stop
        bad_lo = drange.start is not None and info.min > drange.start
        if np.any(bad_hi) or np.any(bad_lo):
            raise ValueError(bad_range_msg)
        self.dtype = dtype

    def copy_with_added_prefix(self: T, prefix=Any) -> T:
        copied = deepcopy(self)
        copied.name = str(prefix) + '_' + copied.name
        return copied

    def __eq__(self, other: 'DomainItem'):
        assert isinstance(other, DomainItem), 'False equivalence (Cannot compare DomainItem to non-DomainItem)'
        return self.name == other.name \
            and self.shape == other.shape \
            and self.dtype == other.dtype \
            and self.range == other.range
