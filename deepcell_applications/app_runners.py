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
"""Helper functions to run Applications"""
import os
import timeit

import numpy as np
import tifffile

import deepcell_applications as dca


def run_application(arg_dict):
    """Takes the user-supplied command line arguments and runs the specified application

    Args:
        arg_dict: dictionary of command line args

    Raises:
        IOError: If specified output file already exists"""
    _ = timeit.default_timer()

    outfile = os.path.join(arg_dict['output_directory'], arg_dict['output_name'])

    # Check that the output path does not exist already
    if os.path.exists(outfile):
        raise IOError(f'{outfile} already exists!')

    app = dca.utils.get_app(arg_dict['app'])

    # load the input image
    image = dca.prepare.prepare_input(arg_dict['app'], **arg_dict)

    # make sure the input image is compatible with the app
    dca.utils.validate_input(app, image)

    # Applications expect a batch dimension
    image = np.expand_dims(image, axis=0)

    # run the prediction
    kwargs = dca.utils.get_predict_kwargs(arg_dict)
    output = app.predict(image, **kwargs)

    # Optionally squeeze the output
    if arg_dict['squeeze']:
        output = np.squeeze(output)

    # save the output as a tiff
    tifffile.imwrite(outfile, output)

    app.logger.info('Wrote output file %s in %s s.',
                    outfile, timeit.default_timer() - _)
