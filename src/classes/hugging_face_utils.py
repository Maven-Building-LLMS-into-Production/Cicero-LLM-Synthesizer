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
Module that includes utilities for interacting with HuggingFace
"""

import logging
import os
from typing import Dict, Optional, Union

import pandas as pd
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi

from src.utils import default_variables as dv

__all__ = ["HuggingFaceHelper"]


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.setLevel(level=logging.INFO)


class HuggingFaceHelper(object):
    """
    Class definition for creating, interacting, and sharing Datasets.
    """

    def __init__(self, **kwargs: Dict) -> None:
        """
        Class definition for creating, interacting, and sharing Datasets.
        """
        # Name of the HuggingFace token as stored in the user's environment
        self.token_name = kwargs.get("token_name", dv.hugging_face_token_name)
        self.username = kwargs.get(
            "username",
            os.environ.get(dv.hugging_face_username_name),
        )

        # HuggingFace endpoint
        self.api_endpoint = "https://huggingface.co"
        self.api = self._authenticate_api()

    def _authenticate_api(self) -> HfApi:
        """
        Method for authenticating with HuggingFace using an authentication
        token.

        Returns
        ---------
        huggingface_api : huggingface_hub.hf_api.HfApi
            Object corresponding to the HuggingFace API after authentication.
        """
        # Check that token is part of the user's environment
        if not os.environ.get(self.token_name):
            msg = f">>> HuggingFace API Token '{self.token_name}' not defined!"
            logger.error(msg)
            raise ValueError(msg)

        # Initializing API object
        return HfApi(
            endpoint=self.api_endpoint,
            token=os.environ.get(self.token_name),
        )

    def convert_dataframe_to_dataset(
        self,
        input_df: pd.DataFrame,
    ) -> Dataset:
        """
        Function to convert an existing DataFrame into a ``Dataset`` object

        Parameters
        -------------
        input_df : pandas.DataFrame
            Variable corresponding to the DataFrame to convert.

        Returns
        -----------
        dataset_obj : datasets.Dataset
            Dataset object with the same data as ``input_df``.
        """

        return Dataset.from_pandas(df=input_df)

    def get_dataset_from_hub(
        self,
        dataset_name: str,
        username: Optional[Union[None, str]] = None,
        split: Optional[Union[None, str]] = None,
    ) -> Dataset:
        # sourcery skip: extract-duplicate-method, use-fstring-for-formatting
        """
        Method for extracting the Dataset from HuggingFace.

        Parameters
        ------------
        dataset_name : str
            Name of the dataset to extract from HuggingFace's Hub.

        username : str, NoneType, optional
            Username to use when extracting the dataset from HuggingFace Hub.
            This variable is set to ``None`` by default.

        split : str, NoneType, optional
            Type of ``split`` to load for the Dataset. If ``None``, the
            method will extract all splits. This variable is set to
            ``None`` by default.

        Returns
        --------
        dataset_obj : datasets.Dataset
            Variable corresponding to the dataset that was extracted
            from the HuggingFace Hub.
        """
        # 'dataset_name' - Type
        dataset_name_type_arr = (str,)
        if not isinstance(dataset_name, dataset_name_type_arr):
            msg = (
                ">> 'dataset_name' ({}) is not a valid input type ({})".format(
                    type(dataset_name),
                    dataset_name_type_arr,
                )
            )
            logger.error(msg)
            raise TypeError(msg)
        # 'username' - Type
        username_type_arr = (str, type(None))
        if not isinstance(username, username_type_arr):
            msg = ">> 'username' ({}) is not a valid input type ({})".format(
                type(username),
                username_type_arr,
            )
            logger.error(msg)
            raise TypeError(msg)
        # 'split' - Type
        split_type_arr = (str, type(None))
        if not isinstance(split, split_type_arr):
            msg = ">> 'split' ({}) is not a valid input type ({})".format(
                type(split),
                split_type_arr,
            )
            logger.error(msg)
            raise TypeError(msg)

        # Defining the path to the dataset in HF.
        dataset_path = (
            f"{username}/{dataset_name}" if username else dataset_name
        )

        return load_dataset(dataset_path, split=split)

    def push_dataset(
        self,
        dataset: Dataset,
        dataset_name: str,
        username: Optional[Union[None, str]] = None,
    ):  # sourcery skip: extract-duplicate-method, use-fstring-for-formatting
        """
        Method for pushing an existing local Dataset to HuggingFace.
        """
        # --- Check input type
        # 'dataset' - Type
        dataset_type_arr = (Dataset,)
        if not isinstance(dataset, dataset_type_arr):
            msg = ">> 'dataset' ({}) is not a valid input type ({})".format(
                type(dataset),
                dataset_type_arr,
            )
            logger.error(msg)
            raise TypeError(msg)
        # 'dataset_name' - Type
        dataset_name_type_arr = (str,)
        if not isinstance(dataset_name, dataset_name_type_arr):
            msg = (
                ">> 'dataset_name' ({}) is not a valid input type ({})".format(
                    type(dataset_name),
                    dataset_name_type_arr,
                )
            )
            logger.error(msg)
            raise TypeError(msg)
        # 'username' - Type
        username_type_arr = (str, type(None))
        if not isinstance(username, username_type_arr):
            msg = ">> 'username' ({}) is not a valid input type ({})".format(
                type(username),
                username_type_arr,
            )
            logger.error(msg)
            raise TypeError(msg)

        # Defining the path to the dataset in HF.
        dataset_path = (
            f"{username}/{dataset_name}" if username else dataset_name
        )

        # Pushing dataset to HuggingFace
        dataset.push_to_hub(
            repo_id=dataset_path,
            token=os.environ.get(self.token_name),
        )
