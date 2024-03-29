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
