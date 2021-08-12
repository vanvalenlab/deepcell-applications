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
"""Tests for deepcell_applications.io"""

import numpy as np

import pytest

import deepcell_applications as dca


def test_load_image(mocker):
    path = 'dummy_path.tiff'

    # test 2D image loaded is cast to 3D properly
    mocker.patch('deepcell_applications.io.get_image',
                 lambda x: np.random.random((32, 32)))
    img = dca.io.load_image(path, channel=0, ndim=3)
    assert img.shape == (32, 32, 1)

    # test 3D image has correct image loaded
    channels = 3
    source = np.random.random((32, 32, channels))
    mocker.patch('deepcell_applications.io.get_image', lambda x: source)
    for c in range(channels):
        img = dca.io.load_image(path, channel=c, ndim=source.ndim)
        assert img.shape == (32, 32, 1)
        np.testing.assert_array_equal(img, source[..., c:c + 1])

    # multiple channels can be selected using a list
    channels = list(range(channels))
    img = dca.io.load_image(path, channel=channels, ndim=source.ndim)
    assert img.shape == (32, 32, 1)
    np.testing.assert_array_equal(img[..., 0], source.sum(axis=-1))

    # test 3D image has correct image loaded (channels first)
    channels = 3
    source = np.random.random((channels, 32, 32))
    mocker.patch('deepcell_applications.io.get_image', lambda x: source)
    for c in range(channels):
        img = dca.io.load_image(path, channel=c, ndim=source.ndim)
        assert img.shape == (32, 32, 1)
        expected = np.expand_dims(source[c], axis=-1)
        np.testing.assert_array_equal(img, expected)

    # multiple channels can be selected using a list
    channels = list(range(channels))
    img = dca.io.load_image(path, channel=channels, ndim=source.ndim)
    assert img.shape == (32, 32, 1)
    np.testing.assert_array_equal(img[..., 0], source.sum(axis=0))

    # test too large of an image fails
    with pytest.raises(ValueError):
        mocker.patch('deepcell_applications.io.get_image',
                     lambda x: np.random.random((30, 32, 32, 1)))
        _ = dca.io.load_image(path, channel=0, ndim=3)

    # test channels out of range throws error
    with pytest.raises(ValueError):
        mocker.patch('deepcell_applications.io.get_image',
                     lambda x: np.random.random((32, 32, 1)))
        _ = dca.io.load_image(path, channel=[0, 4], ndim=3)

    # Test invalid (falsey) values raise IOError
    bad_values = [None, '', False]
    for bad_value in bad_values:
        with pytest.raises(IOError):
            dca.io.load_image(bad_value)
