import numpy as np
from domain.conversion import FeatureConversions
from domain.features import CoordinateFeature


def test_feature_conversions_coordinate_as_indexed():
    # Zero-indexed coordinate
    feature = CoordinateFeature("test1", [0, 0, 0], [5, 5, 5], is_discrete=True)
    assert feature.region_dims == [5, 5, 5]

    sliced_data = np.array([2, 3, 4])
    onehot = FeatureConversions.as_onehot(feature, sliced_data)

    onehot_converted_target = np.zeros([5, 5, 5])
    onehot_converted_target[2, 3, 4] = 1
    onehot_converted_target = onehot_converted_target.ravel()
    assert all(onehot == onehot_converted_target)

    index = FeatureConversions.as_index(feature, sliced_data)
    index_converted_target = onehot_converted_target.ravel().argmax()
    assert index_converted_target == index

    # Offset-indexed coordinate
    feature = CoordinateFeature("test2", [0, -1, 5, 3, 100], [3, 0, 6, 6, 105], is_discrete=True)
    sliced_data = np.array([2, -1, 5, 4, 102])
    onehot = FeatureConversions.as_onehot(feature, sliced_data)

    assert feature.region_dims == [3, 1, 1, 3, 5]

    onehot_converted_target = np.zeros([3, 1, 1, 3, 5])
    onehot_converted_target[tuple(np.subtract([2, -1, 5, 4, 102], [0, -1, 5, 3, 100]))] = 1
    onehot_converted_target = onehot_converted_target.ravel()
    assert all(onehot == onehot_converted_target)

    index = FeatureConversions.as_index(feature, sliced_data)
    index_converted_target = onehot_converted_target.ravel().argmax()
    assert index_converted_target == index


if __name__ == '__main__':
    test_feature_conversions_coordinate_as_indexed()
