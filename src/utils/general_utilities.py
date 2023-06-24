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
Module that includes general utitlity functions.
"""

import argparse as argparse
import logging as logging
import re
from argparse import ArgumentParser as _ArgumentParser
from argparse import HelpFormatter as _HelpFormatter
from operator import attrgetter as _attrgetter
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.setLevel(level=logging.INFO)


__all__ = ["get_project_paths"]


def _get_root_dir():
    """
    Function for determining the path to the root directory of the project.

    Returns
    ----------
    root_dir : str
        Path to the root directory of the project.
    """

    return str(list(Path(__file__).resolve().parents)[2].resolve())


def get_project_paths() -> Dict[str, Path]:
    """
    Function to extract the set of directories of the project.

    Returns
    ----------
    proj_dict : dict
        Dictionary containing the path to the project's directories.
    """
    # --- Defining set of directories
    # Base directory
    base_dir = Path(_get_root_dir())
    # Data directory
    data_dir = base_dir.joinpath("data").resolve()
    # Source directory / Codebase
    src_dir = base_dir.joinpath("src").resolve()

    # --- Creating project dictionary with the project directories
    proj_dict = {
        "base": base_dir,
        "data": data_dir,
        "src": src_dir,
    }

    # --- Making sure the directories exist
    for directory in proj_dict.values():
        directory.mkdir(
            exist_ok=True,
            parents=True,
        )

    return proj_dict


def is_float(s: str):
    """
    Function that checks whether or not ``s` is a string.
    """
    return s.count(".") == 1 and s.replace(".", "").isdigit()


def _str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class SortingHelpFormatter(_HelpFormatter):
    def add_arguments(self, actions):
        """
        Modifier for `argparse` help parameters, that sorts them alphabetically
        """
        actions = sorted(actions, key=_attrgetter("option_strings"))
        super(SortingHelpFormatter, self).add_arguments(actions)


def _get_parser_obj(description: str):
    """
    Function to create an 'argparse' ``parser`` object.
    """

    return _ArgumentParser(
        description=description,
        formatter_class=SortingHelpFormatter,
    )


def show_params(
    params_dict: Dict,
    logger: logging.Logger,
    columns_to_omit: Optional[Union[List, None]] = None,
):
    """
    Function to show the defined of the class.
    """
    # Checking input parameters
    columns_to_omit = columns_to_omit or []
    #
    msg = "\n" + "-" * 50 + "\n"
    msg += "\t---- INPUT PARAMETERS ----" + "\n"
    msg += "" + "\n"
    # Sorting keys of dictionary
    keys_sorted = np.sort(list(params_dict.keys()))
    for key_ii in keys_sorted:
        if key_ii not in columns_to_omit:
            msg += f"\t>>> {key_ii} : {params_dict[key_ii]}\n"
    #
    msg += "\n" + "-" * 50 + "\n"
    logger.info(msg)

    return


def check_url_or_file_type(object_path: str) -> str:
    """
    Function to determine whether the input variable is a file or a URL.

    Parameters
    ------------
    object_path : str
        Path to the object.

    Returns
    ------------
    object_type : str
        Type of the object.

        Options :
            - `url` : The object is a valid URL
            - `file` : The object corresponds to a local file.
            - `unspecified` : This object is neither a file nor a URL.
    """
    # Set of regular expressions for each type
    url_pattern = r"^https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+/?\S*$"

    if re.match(url_pattern, object_path):
        return "url"

    # Checking if 'object_path' is a file or directory
    return "file" if Path(object_path).is_file() else "unspecified"
