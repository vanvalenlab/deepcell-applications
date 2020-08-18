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
"""An example script for Multiplex Segmentation using deepcell.applications."""

import argparse
import os

from skimage.external import tifffile

import deepcell
from deepcell.utils import get_image
from deepcell.applications import MultiplexSegmentation


def get_arg_parser():
    """argument parser to consume command line arguments"""
    parser = argparse.ArgumentParser()

    # Model Config args
    parser.add_argument('infile', help='Image file to process.')

    return parser


def run(infile, image_mpp=0.5, compartment='whole-cell'):
    assert os.path.isfile(infile)

    # construct the output file path (append "_output" to the infile name)
    outfile = os.path.join(
        os.path.dirname(infile),
        '{}_output.tif'.format(os.path.splitext(os.path.basename(infile))[0])
    )

    # load the infile into a numpy array
    img = deepcell.utils.get_image(infile)

    # create the multiplex segmentation
    app = MultiplexSegmentation()

    # run the prediction
    output = app.predict(img, image_mpp=image_mpp, compartment=compartment)

    # save the output as a tiff
    # TODO: channels first/last? What is desired output?
    tifffile.imsave(outfile, output)

    return outfile


if __name__ == '__main__':
    ARGS = get_arg_parser().parse_args()

    OUTFILE = run(ARGS.infile)

    print(OUTFILE)
