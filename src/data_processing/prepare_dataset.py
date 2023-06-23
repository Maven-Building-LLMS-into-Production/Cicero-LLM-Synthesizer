# MIT License
#
# Copyright (c) 2023 Victor Calderon
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Module for preparing the input dataset.
"""

import logging
from pathlib import Path
from typing import Dict

from src.classes import data_preparation as dp
from src.utils import default_variables as dv
from src.utils import general_utilities as gu

__author__ = ["Victor Calderon"]
__copyright__ = ["Copyright 2023 Victor Calderon"]
__all__ = []

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
)
logger.setLevel(logging.INFO)


# ---------------------------- PROJECT VARIABLES ------------------------------

MODULE_DESCRIPTION = "Module for data preparation"
MODULE_VERSION = "1.0"


# ----------------------------- INPUT PARAMETERS ------------------------------


def get_parser():
    """
    Function to get the input parameters to the script.
    """
    # Defining the 'parser' object to use
    parser = gu._get_parser_obj(description=MODULE_DESCRIPTION)

    # Path to the input dataset
    parser.add_argument(
        "--dataset-path",
        dest="dataset_path",
        default=dv.cicero_dataset_url,
        type=str,
        help="""
        Path / URL to the input dataset.
        [Default: '%(default)s']
        """,
    )

    return parser.parse_args()


# -------------------------------  FUNCTIONS ----------------------------------


def _resolve_input_object_path(object_path: str) -> str:
    """
    Check whether or not the path corresponds to a local file or a URL.

    Parameters
    -------------
    object_path : str
        Path of the input object.

    Returns
    ----------
    parsed_object_path : str
        Modified / parsed version of the input object ``object_path``.

    Raises
    ------------
    TypeError ; Error
        This error gets raised whenever the input object is neither
        a 'file' nor a valid 'url'.
    """
    object_type = gu.check_url_or_file_type(object_path=object_path)

    if object_type == "unspecified":
        msg = (
            f">>> Unspecified data type for '{object_path}' or does not exist"
        )
        logger.error(msg)
        raise TypeError(msg)

    return (
        object_path
        if object_type == "url"
        else str(Path(object_path).resolve())
    )


# ------------------------------ MAIN FUNCTIONS -------------------------------


def main(params_dict: Dict):
    """
    Main function to process the data.
    """
    # Determine if the path corresponds to a file or a URL
    params_dict["object_path"] = _resolve_input_object_path(
        params_dict["dataset_path"]
    )

    # Showing set of input parameters
    gu.show_params(params_dict=params_dict, logger=logger)

    # Initializing input parameters
    data_prep_obj = dp.DatasetPrep(dataset_path=params_dict["object_path"])
    data_prep_obj.show_params()
    clean_dataset = data_prep_obj.clean_dataset()

    logger.info(f"\n>>> Raw dataset: \n{data_prep_obj.raw_dataset}\n")
    logger.info(f"\n>>> Clean dataset: \n{clean_dataset}\n")

    return


if __name__ == "__main__":
    # Getting input parameters
    params_dict = vars(get_parser())
    # Running main function
    main(params_dict=params_dict)
