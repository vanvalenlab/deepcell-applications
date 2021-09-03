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
"""Tests for deepcell_applications.argparse"""

import argparse
import os
import stat
import tempfile


import numpy as np
import skimage.io as io

import pytest

import deepcell_applications as dca


def test_get_arg_parser_mesmer():
    with tempfile.TemporaryDirectory() as temp_dir:
        img = np.zeros((10, 10))

        # generate good paths
        file_path = os.path.join(temp_dir, 'img.tiff')
        dir_path = os.path.join(temp_dir, 'dir')

        io.imsave(file_path, img)
        os.makedirs(dir_path)
        io.imsave(dir_path + 'img2.tiff', img)

        # generate bad paths
        bad_file_path = os.path.join(temp_dir, 'fake_img.tiff')
        bad_dir_path = os.path.join(temp_dir, 'fake_dir')

        # make directory read only
        read_dir_path = os.path.join(temp_dir, 'read_dir')
        os.makedirs(read_dir_path)
        os.chmod(read_dir_path, stat.S_IREAD)

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

        ARGS = dca.argparse.get_arg_parser().parse_args(input_list)

        args_dict = vars(ARGS)

        # Mac OS system temp folder starts with /private
        if args_dict['output_directory'].startswith('/private'):
            args_dict['output_directory'] = args_dict['output_directory'].split('/private')[1]

        assert args_dict == output_dict

        # Best way to catch argparse errors https://stackoverflow.com/a/5943381
        with pytest.raises(SystemExit):
            # bad nuclear image path
            _ = dca.argparse.get_arg_parser().parse_args([output_dict['app'],
                                                          '--nuclear-image', bad_file_path,
                                                          '--output-directory', dir_path])

        with pytest.raises(SystemExit):
            # bad membrane image path
            _ = dca.argparse.get_arg_parser().parse_args([output_dict['app'],
                                                          '--nuclear-image', file_path,
                                                          '--membrane-image', bad_file_path,
                                                          '--output-directory', dir_path])

        with pytest.raises(argparse.ArgumentTypeError):
            # bad output dir
            _ = dca.argparse.get_arg_parser().parse_args([output_dict['app'],
                                                          '--nuclear-image', file_path,
                                                          '--output-directory', bad_dir_path])
        with pytest.raises(argparse.ArgumentTypeError):
            # can't write to output dir
            _ = dca.argparse.get_arg_parser().parse_args([output_dict['app'],
                                                          '--nuclear-image', file_path,
                                                          '--output-directory', read_dir_path])
