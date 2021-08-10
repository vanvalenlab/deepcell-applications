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
"""An example script for Multiplex Segmentation using deepcell.applications."""

import argparse
import logging
import os
import sys
import timeit

import numpy as np
import tifffile

import deepcell

import dca
from dca.io import load_image


def get_arg_parser():
    """argument parser to consume command line arguments"""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()

    # first, the arguments that are common to all applications

    class WritableDirectoryAction(argparse.Action):
        # Check that the provided output directory exists, and is writable
        # From: https://gist.github.com/Tatsh/cc7e7217ae21745eb181
        def __call__(self, parser, namespace, values, option_string=None):
            prospective_dir = values
            if not os.path.isdir(prospective_dir):
                raise argparse.ArgumentTypeError(
                    '{} is not a valid directory'.format(
                        prospective_dir,))
            if os.access(prospective_dir, os.W_OK | os.X_OK):
                setattr(namespace, self.dest, os.path.realpath(prospective_dir))
                return
            raise argparse.ArgumentTypeError(
                '{} is not a writable directory'.format(
                    prospective_dir))

    def existing_file(x):
        if x is not None and not os.path.exists(x):
            raise argparse.ArgumentTypeError('{} does not exist.'.format(x))
        return x

    parser.add_argument('--output-directory', '-o',
                        default=os.path.join(root_dir, 'output'),
                        action=WritableDirectoryAction,
                        help='Directory where application outputs are saved.')

    parser.add_argument('--output-name', '-f', required=False,
                        default='mask.tif',
                        help='Name of output file.')

    parser.add_argument('--mpp', type=float, default=0.5,
                        help='Input image resolution in microns-per-pixel.')

    parser.add_argument('-L', '--log-level', default='DEBUG',
                        choices=('DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL'),
                        help='Only log the given level and above.')

    # use subparsers to group options for different applications
    # https://stackoverflow.com/a/30217387
    subparsers = parser.add_subparsers(dest='app', help='application name')

    # Mesmer options configuration:
    mesmer_parser = subparsers.add_parser('mesmer', help='Run Mesmer')

    # Image file inputs
    mesmer_parser.add_argument('--nuclear-image', '-n', required=True,
                               type=existing_file,
                               help=('Path to 2D single channel TIF file.'))

    mesmer_parser.add_argument('--nuclear-channel', '-nc', default=0, type=int,
                               help='Channel to use of the nuclear image.')

    mesmer_parser.add_argument('--membrane-image', '-m', required=False,
                               type=existing_file,
                               help=('Path to 2D single channel TIF file. '
                                     'Optional. If not provided, membrane '
                                     'channel input to network is blank.'))

    mesmer_parser.add_argument('--membrane-channel', '-mc', default=0, type=int,
                               help='Channel to use of the membrane image.')

    # Inference parameters
    mesmer_parser.add_argument('--compartment', '-c', default='whole-cell',
                        choices=('nuclear', 'membrane', 'whole-cell'),
                        help=('The cellular compartment to segment.'))

    return parser


def initialize_logger(log_level):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    log_level = getattr(logging, log_level)

    fmt = '[%(asctime)s]:[%(levelname)s]:[%(name)s]: %(message)s'
    formatter = logging.Formatter(fmt=fmt)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(log_level)
    logger.addHandler(console)


def prepare_mesmer_input(nuclear_path, membrane_path=None, ndim=3,
                         nuclear_channel=0, membrane_channel=0):
    """Load and reshape image input files for the Mesmer application

    Args:
        nuclear_path (str): The path to the nuclear image file
        membrane_path (str): The path to the membrane image file
        ndim (int): Rank of the expected image size
        nuclear_channel (int): The channel of the nuclear data,
            if the image includes a channel axis.
        membrane_channel (int): The channel of the membrane data,
            if the image includes a channel axis.

    Returns:
        numpy.array: Single array of input images concatenated on channels.
    """
    # load the input files into numpy arrays
    nuclear_img = load_image(nuclear_path, channel=nuclear_channel, ndim=ndim)

    # membrane image is optional
    if membrane_path:
        membrane_img = load_image(membrane_path, channel=membrane_channel, ndim=ndim)
    else:
        membrane_img = np.zeros(nuclear_img.shape, dtype=nuclear_img.dtype)

    # join the inputs in the correct order
    img = np.concatenate([nuclear_img, membrane_img], axis=-1)

    return img


if __name__ == '__main__':
    _ = timeit.default_timer()

    ARGS = get_arg_parser().parse_args()

    initialize_logger(log_level=ARGS.log_level)

    OUTFILE = os.path.join(ARGS.output_directory, ARGS.output_name)

    # Check that the output path does not exist already
    if os.path.exists(OUTFILE):
        raise IOError(f'{OUTFILE} already exists!')

    app = dca.runner.get_app(ARGS.app)

    # load the input image
    if ARGS.app == 'mesmer':
        # TODO: move into more parameterizable
        image = prepare_mesmer_input(
            nuclear_path=ARGS.nuclear_image,
            membrane_path=ARGS.membrane_image,
            ndim=len(app.model_image_shape),
            nuclear_channel=ARGS.nuclear_channel,
            membrane_channel=ARGS.membrane_channel,
        )

    else:
        raise ValueError('invalid app: %s' % ARGS.app)

    # make sure the input image is compatible with the app
    dca.runner.validate_input(app, image, ARGS.app)

    # Applications expect a batch dimension
    image = np.expand_dims(image, axis=0)

    # run the prediction
    kwargs = dca.runner.get_predict_kwargs(ARGS)
    output = app.predict(image, **kwargs)

    # save the output as a tiff
    tifffile.imsave(OUTFILE, output)

    app.logger.info('Wrote output file %s in %s s.',
                    OUTFILE, timeit.default_timer() - _)