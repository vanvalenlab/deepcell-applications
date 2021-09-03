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
"""Tests for deepcell_applications.utils"""

import copy

import numpy as np

import pytest

import deepcell_applications as dca


class DummyApplication(object):

    def __init__(self, *args, **kwargs):
        self.model_image_shape = (32, 32, 1)

    @property
    def required_channels(self):
        return self.model_image_shape[-1]


MOCKED_APPLICATIONS = {
    'dummyapplication': {
        'class': DummyApplication,
        'predict_options': ['test'],
    }
}


def test_get_app(mocker):
    # mock the application config in imported settings
    mocker.patch('deepcell_applications.settings.VALID_APPLICATIONS',
                 MOCKED_APPLICATIONS)

    key = list(MOCKED_APPLICATIONS.keys())[0]
    app = dca.utils.get_app(key)
    assert isinstance(app, DummyApplication)

    # test case insensitive
    app2 = dca.utils.get_app(key.upper())
    assert isinstance(app2, DummyApplication)
    assert app2.required_channels == app.required_channels
    app3 = dca.utils.get_app(key.lower())
    assert isinstance(app3, DummyApplication)
    assert app2.required_channels == app3.required_channels

    with pytest.raises(ValueError):
        _ = dca.utils.get_app('bad_app_name')


def test_validate_input():
    app = DummyApplication()

    # x and y don't matter, but rank and num channels do.
    good_combinations = [
        ((32, 32, 1), (64, 64, 1)),
        ((32, 32, 1), (16, 64, 1)),
        ((32, 32, 1), (16, 16, 1)),
        ((16, 16, 3), (32, 32, 3)),
        ((32, 32, 3), (32, 32, 3)),
        ((64, 64, 3), (32, 32, 3)),
    ]
    for model_image_shape, image_shape in good_combinations:
        app.model_image_shape = model_image_shape
        dca.utils.validate_input(app, np.random.random(image_shape))

    bad_combinations = [
        # rank too small (no channels)
        ((32, 32, 1), (32, 32)),
        ((32, 32, 1), (64, 64)),
        ((32, 32, 1), (16, 16)),
        ((32, 32, 1), (16, 64)),
        # rank too large
        ((32, 32, 1), (1, 32, 32, 1)),
        ((32, 32, 1), (1, 32, 32, 1)),
        ((32, 32, 1), (1, 1, 32, 32, 1)),
        # wrong channels
        ((32, 32, 1), (32, 32, 3)),
        ((32, 32, 3), (32, 32, 1)),
    ]
    for model_image_shape, image_shape in bad_combinations:
        app.model_image_shape = model_image_shape
        with pytest.raises(ValueError):
            dca.utils.validate_input(app, np.random.random(image_shape))


def test_get_predict_kwargs(mocker):
    # mock the application config in imported settings
    mocker.patch('deepcell_applications.settings.VALID_APPLICATIONS',
                 MOCKED_APPLICATIONS)

    key = list(MOCKED_APPLICATIONS.keys())[0]

    mock_namespace = {
        'app': key,
        'test': True
    }

    predict_kwargs = dca.utils.get_predict_kwargs(mock_namespace)
    assert predict_kwargs == {'test': True}

    # passed a bad name as `app`
    with pytest.raises(ValueError):
        bad_namespace = copy.copy(mock_namespace)
        bad_namespace['app'] = 'bad_name'
        _ = dca.utils.get_predict_kwargs(bad_namespace)

    # the argparser is misconfigured and not providing a required value
    with pytest.raises(KeyError):
        bad_namespace = copy.copy(mock_namespace)
        del bad_namespace['test']
        _ = dca.utils.get_predict_kwargs(bad_namespace)


