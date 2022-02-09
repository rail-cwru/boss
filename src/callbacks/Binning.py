import numpy as np
from . import Callback
from controller import MDPController

# TODO get rid of in some way shape or form


class Binning(Callback):
    """Convert continuous actions to discrete actions."""
    # TODO init time callback needs to be thought through better.

    def __init__(self, controller: MDPController):
        pass

    @staticmethod
    def normalize(val, min_val, max_val):
        """Normalize the value.

        Normalize the value using a Gaussian distribution.
        """
        return (float(val) - min_val) / (float(max_val) - min_val)

    def discretize(self, data, n_bins):
        """Discritizes the data.

        Converts a continuous range of values to discrete values based on the
        range and bin_count.

        Params:
        -------
        data: np.array
            The data to be discritized.

        n_bins: int
            The number of bins to split the data into. Usually, the higher number of bins the better.
        """
        min_val, max_val = np.min(data), np.max(data)
        bin_width = (max_val - min_val) / n_bins
        # TODO: Should use regular numpy as this is an extra dependency that slows down
        # return pd.cut(data, bins=n_bins)
        raise NotImplementedError()

    def continuous(self, data):
        """Converts discrete data back to the continuous range.

        Params:
        -------
        data: np.array
            Discritized data that needs to be converted back to continous range.
        """
        data = np.array(data)
        n_bins = len(np.unique(data))
        min_val, max_val = np.min(data), np.max(data)
        bin_width = (max_val - min_val) / n_bins
        return data * bin_width
