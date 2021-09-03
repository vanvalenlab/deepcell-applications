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
"""Top level script to run Applications."""
import logging
import os
import sys

from deepcell_applications.argparse import get_arg_parser
from deepcell_applications.app_runners import run_application


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

    # get command line args
    ARGS = get_arg_parser().parse_args()

    OUTFILE = os.path.join(ARGS.output_directory, ARGS.output_name)

    initialize_logger(log_level=ARGS.log_level)

    # run application
    run_application(dict(ARGS._get_kwargs()))
