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
import timeit

from skimage.external import tifffile

import numpy as np

import deepcell


def get_arg_parser():
    """argument parser to consume command line arguments"""
    parser = argparse.ArgumentParser()

    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Image file inputs
    parser.add_argument('--nuclear-image', '-n', required=False,
                        help=('Path to 2D single channel TIF file.'))

    parser.add_argument('--membrane-image', '-m', required=False,
                        help=('Path to 2D single channel TIF file. Optional. '
                              'If not provided, membrane channel input to '
                              'network is blank.'))

    # Mask outputs
    parser.add_argument('--output-directory', '-o',
                        default=os.path.join(root_dir, 'output'),
                        help='Directory where segmentation masks are saved.')

    parser.add_argument('--output-name', '-f', required=False,
                        default='mask.tif',
                        help='Name of output file.')

    # Inference parameters
    parser.add_argument('--compartment', '-c', default='whole-cell',
                        choices=('nuclear', 'whole-cell'),
                        help=('Passed as argument to the '
                              'MultiplexSegmentation application.'))

    parser.add_argument('--mpp', type=float, default=0.5,
                        help='Input image microns-per-pixel.')

    return parser


def run(outpath, nuclear_path, membrane_path=None, image_mpp=0.5, compartment='whole-cell'):
    # load the input files into numpy arrays
    nuclear_img = deepcell.utils.get_image(nuclear_path)
    nuclear_img = np.expand_dims(nuclear_img, axis=-1)

    if membrane_path is not None:
        membrane_img = deepcell.utils.get_image(membrane_path)
        membrane_img = np.expand_dims(membrane_img, axis=-1)
    else:
        membrane_img = np.zeros(nuclear_img.shape, dtype=nuclear_img.dtype)

    img = np.concatenate([nuclear_img, membrane_img], axis=-1)

    # validate correct shape of image
    if len(img.shape) != 3:
        raise ValueError('Invalid image shape. An image of shape {} was '
                         'supplied, but the multiplex model expects of images '
                         'of shape [height, widths, 2]'.format(img.shape))

    # multi-channel tifs render better as channels first in ImageJ
    if img.shape[0] == 2:
        img = np.rollaxis(img, 0, 3)

    elif img.shape[2] != 2:
        raise ValueError('Invalid image shape. An image of shape {} was supplied, '
                         'but the multiplex model expects images of shape'
                         '[height, widths, 2]'.format(img.shape))

    # Applications expect a batch dimension
    img = np.expand_dims(img, axis=0)

    # create the multiplex segmentation
    app = deepcell.applications.MultiplexSegmentation()

    # run the prediction
    output = app.predict(img, image_mpp=image_mpp, compartment=compartment)

    # save the output as a tiff
    tifffile.imsave(outpath, output)


if __name__ == '__main__':
    _ = timeit.default_timer()

    ARGS = get_arg_parser().parse_args()

    # Check that the provided output directory exists, and is writable
    if not os.path.isdir(ARGS.output_directory):
        raise IOError(f'{ARGS.output_directory} is not a directory.')
    if not os.access(ARGS.output_directory, os.W_OK | os.X_OK):
        raise IOError(f'{ARGS.output_directory} is not writable.')

    OUTFILE = os.path.join(ARGS.output_directory, ARGS.output_name)

    # Check that the output path does not exist already
    if os.path.exists(OUTFILE):
        raise IOError(f'{OUTFILE} already exists!')

    # Check that the input files exist
    if not os.path.exists(ARGS.nuclear_image):
        raise IOError(f'{ARGS.nuclear_image} does not exist!')

    run(
        outpath=OUTFILE,
        nuclear_path=ARGS.nuclear_image,
        membrane_path=ARGS.membrane_image,
        image_mpp=ARGS.mpp,
        compartment=ARGS.compartment
    )

    print('Wrote output file {} in {} s.'.format(
        OUTFILE, timeit.default_timer() - _))
