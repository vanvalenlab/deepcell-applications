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
import tempfile

import skimage.io as io
import numpy as np

import deepcell_applications as dca


def test_run_app_mesmer(mocker):
    with tempfile.TemporaryDirectory() as temp_dir:
        # create required input files and directories
        img = np.zeros((10, 10))
        output_dir = os.path.join(temp_dir, 'output_dir')
        img_path = os.path.join(temp_dir, 'img.tiff')

        io.imsave(img_path, img)
        os.makedirs(output_dir)

        # patch user supplied arguments
        class mocked_get_args_mesmer():
            def parse_args(self):
                required_inputs = ['mesmer',
                                   '--output-directory', output_dir,
                                   '--nuclear-image', img_path
                                   ]
                formatted_args = dca.argparse.get_arg_parser().parse_args(required_inputs)
                return formatted_args

        mocker.patch('deepcell_applications.app_runners.get_arg_parser',
                     mocked_get_args_mesmer)

        dca.app_runners.run_application()
