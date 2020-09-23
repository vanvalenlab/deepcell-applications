# Copyright 2016-2020 The Van Valen Lab at the California Institute of
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
"""Tests for main"""
import os

import numpy as np
import skimage.io as io

import main


def test_run(tmp_dir):
    nuc_img = np.random.randint(0, 30, (30, 30))
    mem_img = np.random.randint(0, 30, (30, 30))

    nuc_path = os.path.join(tmp_dir, 'nuc_img.tiff')
    mem_path = os.path.join(tmp_dir, 'mem_img.tiff')
    io.imsave(nuc_path, nuc_img)
    io.imsave(mem_path, mem_img)

    output_path = os.path.join(tmp_dir, 'output_image.tiff')

    main.run(outpath=output_path, nuclear_path=nuc_path, membrane_path=mem_path)

    assert os.path.exists(output_path)
