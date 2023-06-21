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

from spacy.lang.en.stop_words import STOP_WORDS

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
        # Lower case the text
        processed_string = self._lowercase_text(input_string)
        # Removing non-alphanumeric characters
        processed_string = self._only_keep_alphanumeric(processed_string)
        # Removing stop words
        processed_string = self._remove_stopwords(processed_string)

        return processed_string


class DatasetPrep(object):
    """
    Class object for the Data Processing of the input dataset.
    """

    def __init__(self, dataset_path: str) -> None:
        """
        Class object for the Data Processing of the input dataset.

        Parameters
        ------------
        dataset_path : str
            Path / URL to the input dataset.
        """
        pass
