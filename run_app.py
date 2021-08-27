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

import deepcell_applications as dca


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


if __name__ == '__main__':
    _ = timeit.default_timer()

    ARGS = get_arg_parser().parse_args()

    initialize_logger(log_level=ARGS.log_level)

    OUTFILE = os.path.join(ARGS.output_directory, ARGS.output_name)

    # Check that the output path does not exist already
    if os.path.exists(OUTFILE):
        raise IOError(f'{OUTFILE} already exists!')

    app = dca.utils.get_app(ARGS.app)

    args_as_kwargs = dict(ARGS._get_kwargs())

    # load the input image
    image = dca.prepare.prepare_input(ARGS.app, **args_as_kwargs)

    # make sure the input image is compatible with the app
    dca.utils.validate_input(app, image)

    # Applications expect a batch dimension
    image = np.expand_dims(image, axis=0)

    # run the prediction
    kwargs = dca.utils.get_predict_kwargs(args_as_kwargs)
    output = app.predict(image, **kwargs)

    # Optinally squeeze the output
    if ARGS.squeeze:
        output = np.squeeze(output)

    # save the output as a tiff
    tifffile.imsave(OUTFILE, output)

    app.logger.info('Wrote output file %s in %s s.',
                    OUTFILE, timeit.default_timer() - _)
