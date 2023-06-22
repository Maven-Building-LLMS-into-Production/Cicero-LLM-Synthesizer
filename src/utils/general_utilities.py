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

import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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


def get_project_paths() -> Dict[Path]:
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

    # --- Creating project dictionary with the project directories
    proj_dict = {
        "base_dir": base_dir,
        "data_dir": data_dir,
    }

    # --- Making sure the directories exist
    for directory in proj_dict.values():
        directory.mkdir(
            exist_ok=True,
            parents=True,
        )

    return proj_dict
