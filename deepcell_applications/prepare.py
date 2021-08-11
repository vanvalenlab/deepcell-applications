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
"""Functions for preparing input data for the applications"""

import numpy as np

import deepcell_applications as dca


def prepare_input(name, **kwargs):
    name = str(name).lower()
    if name == 'mesmer':
        return prepare_mesmer_input(**kwargs)
    else:
        raise ValueError('Invalid application name: {}'.format(name))


def prepare_mesmer_input(nuclear_path, membrane_path=None, ndim=3,
                         nuclear_channel=0, membrane_channel=0, **kwargs):
    """Load and reshape image input files for the Mesmer application

    Args:
        nuclear_path (str): The path to the nuclear image file
        membrane_path (str): The path to the membrane image file
        ndim (int): Rank of the expected image size
        nuclear_channel (list): Integer or list of integers for the relevant
            nuclear channels of the nuclear image data.
            All channels will be summed into a single tensor.
        membrane_channel (int): Integer or list of integers for the relevant
            nuclear channels of the membrane image data.
            All channels will be summed into a single tensor.

    Returns:
        numpy.array: Single array of input images concatenated on channels.
    """
    # load the input files into numpy arrays
    nuclear_img = dca.io.load_image(
        nuclear_path,
        channel=nuclear_channel,
        ndim=ndim)

    # membrane image is optional
    if membrane_path:
        membrane_img = dca.io.load_image(
            membrane_path,
            channel=membrane_channel,
            ndim=ndim)
    else:
        membrane_img = np.zeros(nuclear_img.shape, dtype=nuclear_img.dtype)

    # join the inputs in the correct order
    img = np.concatenate([nuclear_img, membrane_img], axis=-1)

    return img
