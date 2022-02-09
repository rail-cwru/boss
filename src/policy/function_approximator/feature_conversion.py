"""
Functions for converting / decoding special semantic features.

Different Function Approximators may call these functions to translate certain
    features into usable forms.

    Each FA should have a default configuration for dealing with the special types.

Since each FA is instantiated for its Policy and thus PolicyGroup,
    they can contain precomputed arrays for the post-conversion data.


Produce a data object that contains instructions on how to remap from
    the source data array into the target data array
    so as to avoid finding the position slices through feature indexing.
"""
# import numpy as np
#
# from domain.base import Domain
# from domain.features import Feature, FeatureType
#
#
#
# # Following functions are meant to be operated on the pre-sliced VIEWS.
#
#
# def encode_coordinate(src: np.ndarray, dst: np.ndarray, target_type: FeatureType, binning=False):
#     if binning:
#         # TODO FUTURE - implement continuous to real binning
#         # Probably have that in a preliminary step
#         raise NotImplementedError()
#
#     if target_type == FeatureType.ONE_HOT:
#         # Only work for discrete coordinate
#         # out = np.zeros(feature.drange.stop - feature.drange.start, dtype='float32')
#
