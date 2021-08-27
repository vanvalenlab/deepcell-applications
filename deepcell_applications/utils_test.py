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
import os

import numpy as np
import skimage.io as io

import argparse
import pytest
import tempfile

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


def test_get_arg_parser():
    with tempfile.TemporaryDirectory() as temp_dir:
        img = np.zeros((10, 10))

        # generate good paths
        file_path = os.path.join(temp_dir, 'img.tiff')
        dir_path = os.path.join(temp_dir, 'dir')

        io.imsave(file_path, img)
        os.makedirs(dir_path)

        # generate bad paths
        bad_file_path = os.path.join(temp_dir, 'fake_img.tiff')
        bad_dir_path = os.path.join(temp_dir, 'fake_dir')

        # create dict to hold expected outputs
        output_dict = {
            'app': 'mesmer',
            'output_directory': dir_path,
            'output_name': 'seg_mask.tif',
            'log_level': 'INFO',
            'squeeze': True,
            'nuclear_path': file_path,
            'nuclear_channel': [2],
            'membrane_path': file_path,
            'membrane_channel': [3],
            'compartment': 'nuclear',
            'image_mpp': 3.0,
            'batch_size': 5}

        # construct syntax for appropriate passing to argparse
        input_list = [output_dict['app'],
                      '--output-directory', output_dict['output_directory'],
                      '--output-name', output_dict['output_name'],
                      '--log-level', output_dict['log_level'],
                      '--squeeze',
                      '--nuclear-image', output_dict['nuclear_path'],
                      '--nuclear-channel', str(output_dict['nuclear_channel'][0]),
                      '--membrane-image', output_dict['membrane_path'],
                      '--membrane-channel', str(output_dict['membrane_channel'][0]),
                      '--compartment', output_dict['compartment'],
                      '--image-mpp', str(int(output_dict['image_mpp'])),
                      '--batch-size', str(output_dict['batch_size'])]

        ARGS = dca.utils.get_arg_parser().parse_args(input_list)

        args_dict = vars(ARGS)
        # argparser adds '/private' to the front of the folder path
        args_dict['output_directory'] = args_dict['output_directory'].split('/private')[1]

        assert args_dict == output_dict

        # I couldn't figure out the correct syntax to catch these errors
        with pytest.raises(SystemExit):
            # bad nuclear image path
            _ = dca.utils.get_arg_parser().parse_args([output_dict['app'],
                                                       '--nuclear-image', bad_file_path,
                                                       '--output-directory', dir_path])

        with pytest.raises(SystemExit):
            # bad membrane image path
            _ = dca.utils.get_arg_parser().parse_args([output_dict['app'],
                                                       '--nuclear-image', file_path,
                                                       '--membrane-image', bad_file_path,
                                                       '--output-directory', dir_path])

        with pytest.raises(argparse.ArgumentTypeError):
            # bad output dir
            _ = dca.utils.get_arg_parser().parse_args([output_dict['app'],
                                                       '--nuclear-image', file_path,
                                                       '--output-directory', bad_dir_path])
