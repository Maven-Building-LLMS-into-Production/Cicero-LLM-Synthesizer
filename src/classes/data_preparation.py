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
Module that contains the class definitions for the data preparation tasks.
"""

import logging
import re
from datetime import datetime
from typing import List, Optional, Tuple, Union

import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS

from src.classes import hugging_face_utils as hf
from src.utils import default_variables as dv
from src.utils import general_utilities as gu

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------- CLASS DEFINITIONS ------------------------------


# -- Defining functions that can be used for cleaning up and preparing text
class NLPPrep(object):
    """
    Class object for handling the data processing of text.
    """

    def __init__(self):
        # Defining the corresponding stop words
        self.stop_words = list(STOP_WORDS)

    def _lowercase_text(self, input_string: str) -> str:
        """
        Method for making the input text lowercase.

        Parameters
        ------------
        input_string : str
            Text variable to lowercase.

        Returns
        ----------
        output_string : str
            Lower-cased version of ``input_string``.
        """

        return input_string.lower()

    def _only_keep_alphanumeric(self, input_string: str) -> str:
        """
        Method for only keeping alphanumerical characters in the text.

        Parameters
        ------------
        input_string : str
            Text variable to filter.

        Returns
        ----------
        output_string : str
            Filtered version of ``input_string`` that only contains
            alphanumerical characters.
        """
        regex_pattern = r"[^a-zA-z0-9\s]"

        return re.sub(regex_pattern, "", input_string)

    def _remove_stopwords(self, input_string: str) -> str:
        """
        Method for removing stop words from the input text.

        Parameters
        ------------
        input_string : str
            Text variable to filter.

        Returns
        ----------
        output_string : str
            Filtered version of ``input_string`` without stop words in
            the text.
        """
        # Splitting the text into 'tokens'
        tokens = input_string.strip().split()

        return " ".join(
            [word for word in tokens if word not in self.stop_words]
        )

    def _remove_unicode(self, input_str: str) -> str:
        """
        Method for removing Unicode from the input text.

        Parameters
        ------------
        input_str : str
            Text variable, from which to remove Unicode characters.

        Returns
        ----------
        string_decode : str
            Filtered version of ``input_str`` without the Unicode characters.
        """
        string_encode = input_str.encode("ascii", "ignore")

        return string_encode.decode()

    def process_text(self, input_string: str) -> str:
        """
        Method for passing the input variable through NLP-based techniques
        to process the text.

        Parameters
        ------------
        input_string : str
            Variable corresponding to the text that will be processed.

        Returns
        ------------
        processed_string : str
            Variable corresponding to the *processed* version of the input
            string, after having gone through some NLP-based processing
            techniques.

        Notes
        -----------
        This function will perform the following NLP-based techniques:

        1. Make the text lowercase.
        2. Remove any non-alphanumeric character from the string.
        3. Remove any stop words from the text.
        """
        # Remove Unicode characters
        processed_string = self._remove_unicode(input_string)
        # Lower case the text
        processed_string = self._lowercase_text(processed_string)
        # Removing non-alphanumeric characters
        processed_string = self._only_keep_alphanumeric(processed_string)
        # Removing stop words
        processed_string = self._remove_stopwords(processed_string)

        return processed_string


class DatasetPrep(object):
    """
    Class object for the Data Processing of the input dataset.
    """

    def __init__(
        self,
        dataset_path: str,
        **kwargs,
    ):
        """
        Class object for the Data Processing of the input dataset.

        Parameters
        ------------
        dataset_path : str
            Path / URL to the input dataset.
        """
        # Path to the output dataset
        self.datasets_dir = gu.get_project_paths()["data"]

        # Other parameters
        for colname in [
            "save_to_disk",
            "document_id_colname",
            "title_colname",
            "content_colname",
            "clean_content_colname",
        ]:
            setattr(self, colname, kwargs.get(colname, getattr(dv, colname)))

        # Initializing dataset
        self.dataset_path = dataset_path
        self.raw_dataset = self._get_dataset()

        # Extracting the number of rows and columns, and column names
        (
            self.n_rows,
            self.n_cols,
            self.columns_names,
        ) = self._get_columns_and_shape()

        # Initializing NLP-Prep Object
        self.nlp_obj = NLPPrep()

    def show_params(self):
        """
        Method for displaying the set of input parameters of the class.
        """

        gu.show_params(
            params_dict=self.__dict__,
            logger=logger,
            columns_to_omit=["raw_dataset"],
        )

    def _get_dataset(self) -> pd.DataFrame:
        # sourcery skip: class-extract-method
        """
        Method for extracting the dataset from the input source.

        Returns
        ----------
        raw_dataset : pandas.DataFrame
            DataFrame containing the data from the input source.
        """
        logger.info(f">> Extracting dataset from `{self.dataset_path}`")

        # Reading in dataset
        raw_dataset = pd.read_csv(self.dataset_path)

        # Saving to disk, if applicable
        if self.save_to_disk:
            dataset_filepath = self.datasets_dir.joinpath("raw_dataset.csv")
            dataset_filepath.parent.mkdir(exist_ok=True, parents=True)
            raw_dataset.to_csv(dataset_filepath, header=True, index=True)

            logger.info(f">> Raw dataset saved to '{str(dataset_filepath)}'")

        return raw_dataset

    def _get_columns_and_shape(self) -> Tuple[int, int, List]:
        # sourcery skip: use-fstring-for-formatting
        """
        Method for extracting the columns and information about the
        raw dataset.

        Returns
        ----------
        n_rows : int
            Number of rows in the original dataset.

        n_cols : int
            Number of columns in the original dataset.

        column_names_arr : list
            List of columns from the original dataset.
        """
        # Number of rows and columns
        n_rows, n_columns = self.raw_dataset.shape

        logger.info(
            ">> There are '{}' rows and '{}' columns in the dataset".format(
                n_rows,
                n_columns,
            )
        )

        # Column names
        column_names_arr = sorted(self.raw_dataset.columns)

        logger.info(
            ">> Columns in the dataset: \n\t{}".format(
                "\n\t".join(column_names_arr)
            )
        )

        return n_rows, n_columns, column_names_arr

    def _process_text(self, input_text: str) -> str:
        """
        Method for applying NLP-based techniques on an input text in order
        to prepare it to be used by the embedding algorithm.

        Parameters
        -----------
        input_text : str
            Variable corresponding to the input text.

        Returns
        -----------
        processed_text : str
            Processed version of the ``input_text``.

        Notes
        ----------
        This function will perform the following NLP-based techniques:

        1. Make the text lowercase.
        2. Remove any non-alphanumeric character from the string.
        3. Remove any stop words from the text.
        """

        return self.nlp_obj.process_text(input_string=input_text)

    def clean_dataset(self) -> pd.DataFrame:
        """
        Method for cleaning the raw dataset and create a clean version
        of the dataset.

        Returns
        ---------
        dataset_clean : pandas.DataFrame
            Clean version of the input dataset, after having gone through
            data-cleaning techniques.
        """
        # --- Start time
        logger.info(">> Data cleaning process ...")
        start_time = datetime.now()
        #

        # --- Making a copy of the raw dataset
        dataset_df = self.raw_dataset.copy()

        # --- Data-cleaning techniques
        # Removing duplicates
        dataset_df.drop_duplicates(keep="first", inplace=True)

        # Removing entries that have 'NaN' in the dataset
        dataset_df.dropna(how="any", inplace=True)

        # Casting proper data types
        dataset_df = dataset_df.astype(str)

        # Resetting the index of the dataset
        dataset_df.reset_index(drop=True, inplace=True)

        # Removing trailing whitespaces
        for colname in [self.document_id_colname, self.title_colname]:
            dataset_df.loc[:, colname] = dataset_df[colname].apply(
                lambda x: x.strip()
            )

        # Processing content
        dataset_df.loc[:, getattr(self, "clean_content_colname")] = dataset_df[
            getattr(self, "content_colname")
        ].apply(lambda text: self.nlp_obj.process_text(text))

        # --- Saving to disk, if applicable
        if self.save_to_disk:
            dataset_filepath = self.datasets_dir.joinpath("clean_dataset.csv")
            dataset_filepath.parent.mkdir(exist_ok=True, parents=True)
            dataset_df.to_csv(dataset_filepath, header=True, index=True)

            logger.info(f">> Clean dataset saved to '{str(dataset_filepath)}'")

        # --- End time
        end_time = datetime.now()
        logger.info(f">>    Finished at: {end_time}")
        logger.info(f">>    Took: {end_time - start_time}")
        logger.info(">> Data cleaning process ... DONE")

        return dataset_df

    def push_dataset_to_hub(
        self,
        dataset: pd.DataFrame,
        dataset_name: str,
        username: Optional[Union[None, str]] = None,
    ):
        """
        Method for pushing the ``dataset`` to the HuggingFace's Hub.

        Parameters
        -------------
        dataset : pandas.DataFrame
            Dataset that will be pushed to HuggingFace.

        dataset_name : str
            Name of the dataset to use.

        username : str, NoneType, optional
            Us
        """
        # Initializing class object
        hf_obj = hf.DatasetHelper()

        # Transforming dataset type
        hf_dataset = hf_obj.convert_dataframe_to_dataset(input_df=dataset)

        # Push dataset to hub
        hf_obj.push_dataset(
            dataset=hf_dataset,
            dataset_name=dataset_name,
            username=username,
        )
