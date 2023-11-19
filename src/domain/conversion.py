import numpy as np

from domain.features import Feature, FeatureType, CoordinateFeature


class FeatureConversions(object):
    """
    A class containing functions for converting feature representations between feature types.

    The intent of these conversion functions is to promote representational interchangeability -
        e.g.
            A Tabular Policy which must discretely represent received states/features
            is unable to natively support a feature of discrete coordinates or continuous values.

            However, discrete coordinates can be mapped to an integer index,
            and continuous values can be binned.

        By providing functions to perform these translations, we allow for interchangeability
            and interoperability.
    """

    @staticmethod
    def as_index(feature: Feature, sliced_data: np.ndarray) -> int:
        """
        Extract the feature from the data as an index.
            i.e. Each distinct values the feature could take on
                 must be mapped to a distinct non-negative integer.

        This is the native representation for:
            FeatureType.DISCRETE
            FeatureType.BINARY

        :param feature: Feature specification
        :param sliced_data: Data containing the data for the feature in its native representation.
        :return: Feature data in unique index representation.
        """
        feature_type = feature.feature_type
        if feature_type == FeatureType.DISCRETE or feature_type == FeatureType.BINARY:
            return sliced_data[0]
        elif feature_type == FeatureType.ONE_HOT:
            return sliced_data.argmax()
        elif feature_type == FeatureType.COORDINATE:
            assert isinstance(feature, CoordinateFeature)
            if not feature.discrete:
                raise TypeError("[{}] is a continuous feature and requires binning to index."
                                .format(feature.name))
            if feature.is_infinite:
                raise TypeError("[{}] is an unbounded feature and cannot be mapped to an index."
                                .format(feature.name))
            if feature.sparse_values:
                for i, value in enumerate(feature.sparse_values):
                    if np.absolute(np.subtract(value, sliced_data)).sum() <= 1e-6:
                        return i
                print('not found', sliced_data)
                raise ValueError("[{}] is a sparse coordinate feature but the value was not in its set."
                                 .format(feature.name))
            else:
                offset = []
                for ind, i in enumerate(sliced_data):
                    offset.append(i - feature.lower[ind])

                # offset = sliced_data - feature.lower
                stride = 1
                idx = 0
                for dim, val in zip(feature.region_dims[::-1], offset[::-1]):
                    idx += stride * val
                    stride *= dim
                return idx
        else:
            raise TypeError("[{}] cannot be mapped to a unique index representation.".format(feature.feature_type))


    @staticmethod
    def as_onehot(feature: Feature, sliced_data: np.ndarray) -> np.ndarray:
        """
        Extract the feature from the data as a one-hot feature.

        :param feature: Feature to extract
        :param sliced_data: Data containing that feature
        :return: Feature data extracted as one-hot
        """

        feature_type = feature.feature_type
        if feature_type == FeatureType.ONE_HOT:
            return sliced_data
        elif feature_type == FeatureType.DISCRETE or feature_type == FeatureType.BINARY:
            x = np.zeros(feature.shape[0])
            x[sliced_data] = 1
            return x
        elif feature_type == FeatureType.COORDINATE:
            assert isinstance(feature, CoordinateFeature)
            if not feature.discrete:
                raise TypeError("[{}] is a continuous feature and cannot be represented as one-hot.")
            if feature.is_infinite:
                raise TypeError("[{}] is an unbounded feature and cannot be mapped to one-hot.")
            if feature.sparse_values:
                for i, value in enumerate(feature.sparse_values):
                    if np.subtract(value, sliced_data).abs().sum() <= 1e-6:
                        x = np.zeros(len(feature.sparse_values))
                        x[i] = 1
                        return x
            else:
                value = np.zeros(feature.num_values(), dtype=feature.dtype)
                value.reshape(feature.region_dims)[tuple(sliced_data - feature.lower)] = 1
                return value
        else:
            raise TypeError("[{}] cannot be mapped to a one-hot representation.".format(feature.feature_type))
