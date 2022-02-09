import numpy as np


class TrainingSamples:
    """Holds inputs and outputs for training"""

    def __init__(self, xs, ys, actions=None, discrete=None):
        """
        Make a dataset (concrete)
        :param xs: In practice, the environment observations
        :param ys: In practice, some rewards measure
        :param actions: An index (discrete) of float (continuous)
        """
        assert(len(xs) == len(ys))
        self.data = [xs, ys]

        if actions is not None:
            assert(len(actions) == len(xs))
            assert(discrete is not None)

            # TODO: It seems like the tf.Tensor is expected [n,2] no matter if the value is discrete or continuous
            if discrete:
                batch_indexes = np.arange(len(actions))
                # TODO: Change if multiple actions are supported
                formatted_indexes = np.array([batch_indexes, actions[:,0]]).T
                self.data.append(formatted_indexes)
            else:  # continuous
                batch_indexes = np.arange(len(actions))
                # TODO: Change if multiple actions are supported
                formatted_indexes = np.array([batch_indexes, actions[:,0]]).T
                self.data.append(formatted_indexes)
                
                #self.data.append(actions) # TODO: This is what was 

    def x(self):
        return self.data[0]

    def y(self):
        return self.data[1]

    def actions(self):
        if len(self.data) == 3:
            return self.data[2]

    def __getattr__(self, idx):
        return self.data[0][idx, :].append(self.data[1][idx, :])
