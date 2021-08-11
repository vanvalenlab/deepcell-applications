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
"""Tests for deepcell_applications.prepare"""

import numpy as np

import pytest

import deepcell_applications as dca


def test_prepare_input():
    # unknown applications fail with ValueError
    with pytest.raises(ValueError):
        dca.prepare.prepare_input('unknown app')

def test_prepare_mesmer_input(mocker):
    # mock the application config in imported settings
    nuclear = np.random.random((32, 32, 1))
    membrane = np.random.random((32, 32, 1))

    def mocked_load_image(path, *_, **__):
        if 'membrane' in str(path):
            return membrane
        return nuclear

    mocker.patch('deepcell_applications.io.load_image',
                 mocked_load_image)

    # test that nuclear image first, then membrane
    img = dca.prepare.prepare_mesmer_input(
        nuclear_path='nuclear',
        membrane_path='membrane',
    )

    np.testing.assert_equal(img[..., 0:1], nuclear)
    np.testing.assert_equal(img[..., 1:2], membrane)

    # no membrane passed should be all zeros
    img = dca.prepare.prepare_mesmer_input(
        nuclear_path='nuclear',
    )

    np.testing.assert_equal(img[..., 0:1], nuclear)
    np.testing.assert_equal(img[..., 1:2], np.zeros_like(nuclear))

    # test that `prepare_input` works
    img = dca.prepare.prepare_input(
        name='mesmer',
        nuclear_path='nuclear',
        membrane_path='membrane',
        unknown_kwarg='test',  # this shouldn't throw error
    )

    np.testing.assert_equal(img[..., 0:1], nuclear)
    np.testing.assert_equal(img[..., 1:2], membrane)
