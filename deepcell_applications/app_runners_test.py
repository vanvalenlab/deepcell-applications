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
"""Tests for deepcell_applications.app_runners"""
import os

import skimage.io as io
import numpy as np

import pytest

import deepcell_applications as dca


def test_run_app_mesmer(tmpdir):
    temp_dir = str(tmpdir)

    # run with default parameters

    # create required input files and directories
    img = np.zeros((10, 10))
    output_dir = os.path.join(temp_dir, 'output_dir')
    img_path = os.path.join(temp_dir, 'img.tiff')

    io.imsave(img_path, img)
    os.makedirs(output_dir)

    required_inputs = ['mesmer',
                       '--output-directory', output_dir,
                       '--nuclear-image', img_path,
                       '--squeeze'
                       ]
    args = dca.argparse.get_arg_parser().parse_args(required_inputs)

    dca.app_runners.run_application(dict(args._get_kwargs()))

    # error checking

    # create required input files and directories
    out_path = os.path.join(temp_dir, 'out_mask.tiff')
    io.imsave(out_path, img)

    # patch user supplied arguments

    required_inputs = ['mesmer',
                       '--output-directory', output_dir,
                       '--nuclear-image', img_path,
                       '--output-name', out_path,
                       '--squeeze'
                       ]
    args_io_error = dca.argparse.get_arg_parser().parse_args(required_inputs)

    with pytest.raises(IOError):
        dca.app_runners.run_application(dict(args_io_error._get_kwargs()))
