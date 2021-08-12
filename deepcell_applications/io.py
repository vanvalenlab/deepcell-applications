# Copyright 2016-2021 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-applications/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functions for reading and writing files."""

import numpy as np

from deepcell.utils.io_utils import get_image


def load_image(path, channel=0, ndim=3):
    """Load an image file as a single-channel numpy array.

    Args:
        path (str): Filepath to the image file to load.
        channel (list): Loads the given channel if available.
            If channel is list of length > 1, each channel
            will be summed.
        ndim (int): The expected rank of the returned tensor.

    Returns:
        numpy.array: The image channel loaded as an array.
    """
    if not path:
        raise IOError('Invalid path: %s' % path)

    img = get_image(path)

    channel = channel if isinstance(channel, (list, tuple)) else [channel]

    # getting a little tricky, which axis is channel axis?
    if img.ndim == ndim:
        # file includes channels, find the channel axis
        # assuming the channels axis is the smallest dimension
        axis = img.shape.index(min(img.shape))
        if max(channel) >= img.shape[axis]:
            raise ValueError('Channel {} was passed but channel axis is '
                             'only size {}'.format(
                                 max(channel), img.shape[axis]))

        # slice out only the required channel
        slc = [slice(None)] * len(img.shape)
        # use integer to select only the relevant channels
        slc[axis] = channel
        img = img[tuple(slc)]
        # sum on the channel axis
        img = img.sum(axis=axis)

    # expand the (proper) channel axis
    img = np.expand_dims(img, axis=-1)

    if not img.ndim == ndim:
        raise ValueError('Expected image with ndim = {} but found ndim={} '
                         'and shape={}'.format(ndim, img.ndim, img.shape))

    return img
