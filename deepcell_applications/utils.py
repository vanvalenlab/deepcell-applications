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
"""Functions for instantiating and running Applications"""

import argparse
import os

import deepcell_applications as dca


def get_app(name, **kwargs):
    """Returns an instantiated Application based on the name.

    Args:
        name (str): The name of the application
        kwargs (dict): Keyword arguments used for application instantiation

    Returns:
        deepcell.applications.Application: The instantiated application
    """
    name = str(name).lower()
    app_map = dca.settings.VALID_APPLICATIONS
    try:
        return app_map[name]['class'](**kwargs)
    except KeyError:
        raise ValueError('{} is not a valid application name. '
                         'Valid applications: {}'.format(
                             name, list(app_map.keys())))


def validate_input(app, img):
    # validate correct shape of image
    rank = len(app.model_image_shape)
    name = app.__class__.__name__
    errtext = ('Invalid image shape. An image of shape {} was provided, but '
               '{} expects of images of shape [height, widths, {}]'.format(
                   img.shape, str(name).capitalize(), app.required_channels))

    if len(img.shape) != len(app.model_image_shape):
        raise ValueError(errtext)

    if img.shape[rank - 1] != app.required_channels:
        raise ValueError(errtext)


def get_predict_kwargs(kwargs):
    """Returns a dictionary for use in ``app.predict``.

    Args:
        kwargs (dict): Parsed command-line arguments.

    Returns:
        dict: The parsed key-value pairs for ``app.predict``.
    """
    name = str(kwargs.get('app')).lower()
    app_map = dca.settings.VALID_APPLICATIONS
    predict_kwargs = dict()
    try:
        app_options = app_map[name]['predict_options']
    except KeyError:
        raise ValueError('{} is not a valid application name. '
                         'Valid applications: {}'.format(
                             name, list(app_map.keys())))
    for k in app_options:
        try:
            predict_kwargs[k] = kwargs[k]
        except KeyError:
            raise KeyError('{} is required for {} jobs, but is not found'
                           'in parsed CLI arguments.'.format(k, name))
    return predict_kwargs


def get_arg_parser():
    """argument parser to consume command line arguments"""
    # root_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = ''
    parser = argparse.ArgumentParser()

    # Create a parent parser for common arguments
    # https://stackoverflow.com/a/63283912
    parent = argparse.ArgumentParser()

    class WritableDirectoryAction(argparse.Action):
        # Check that the provided output directory exists, and is writable
        # From: https://gist.github.com/Tatsh/cc7e7217ae21745eb181
        def __call__(self, parser, namespace, values, option_string=None):
            prospective_dir = values
            if not os.path.isdir(prospective_dir):
                raise argparse.ArgumentTypeError(
                    '{} is not a valid directory'.format(
                        prospective_dir, ))
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

    parent.add_argument('--output-directory', '-o',
                        default=os.path.join(root_dir, 'output'),
                        action=WritableDirectoryAction,
                        help='Directory where application outputs are saved.')

    parent.add_argument('--output-name', '-f',
                        default='mask.tif',
                        help='Name of output file.')

    parent.add_argument('-L', '--log-level', default='INFO',
                        choices=('DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL'),
                        help='Only log the given level and above.')

    parent.add_argument('--squeeze', action='store_true',
                        help='Squeeze the output tensor before saving.')

    # use subparsers to group options for different applications
    # https://stackoverflow.com/a/30217387
    subparsers = parser.add_subparsers(dest='app', help='application name')

    # Next, each application is configured as its own subparser
    # Each subparser should inherit from ``parent`` to inherit options
    # Configurable inputs for ``prepare_input`` and ``app.predict`` must
    # match (via name or dest) the function input names exactly.

    # Mesmer Application Configuration
    mesmer = subparsers.add_parser('mesmer', parents=[parent], add_help=False,
                                   help='Run Mesmer on nuclear + membrane data')

    # Mesmer Image file inputs
    mesmer.add_argument('--nuclear-image', '-n', required=True,
                        type=existing_file, dest='nuclear_path',
                        help=('REQUIRED: Path to 2D single channel TIF file.'))

    mesmer.add_argument('--nuclear-channel', '-nc',
                        default=0, nargs='+', type=int,
                        help='Channel(s) to use of the nuclear image. '
                             'If more than one channel is passed, '
                             'all channels will be summed.')

    mesmer.add_argument('--membrane-image', '-m',
                        type=existing_file, dest='membrane_path',
                        help=('Path to 2D single channel TIF file. '
                              'Optional. If not provided, membrane '
                              'channel input to network is blank.'))

    mesmer.add_argument('--membrane-channel', '-mc',
                        default=0, nargs='+', type=int,
                        help='Channel(s) to use of the membrane image. '
                             'If more than one channel is passed, '
                             'all channels will be summed.')

    # Mesmer Inference parameters
    mesmer.add_argument('--image-mpp', type=float, default=0.5,
                        help='Input image resolution in microns-per-pixel. '
                             'Default value of 0.5 corresponds to a 20x zoom.')

    mesmer.add_argument('--batch-size', '-b', default=4, type=int,
                        help='Batch size for `model.predict`.')

    mesmer.add_argument('--compartment', '-c', default='whole-cell',
                        choices=('nuclear', 'membrane', 'whole-cell'),
                        help=('The cellular compartment to segment.'))

    return parser
