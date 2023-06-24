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

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer

from src.utils import default_variables as dv

__author__ = ["Victor Calderon"]
__copyright__ = ["Copyright 2023 Victor Calderon"]
__all__ = ["SemanticSearchEngine"]

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
)
logger.setLevel(logging.INFO)

# --------------------------- CLASS DEFINITIONS -------------------------------


class SemanticSearchEngine(object):
    """
    Class object for running Semantic Search on the input dataset.
    """

    def __init__(self, **kwargs):
        """
        Class object for running Semantic Search on the input dataset.
        """
        # --- Defining variables
        # Device to use, i.e. CPU or GPU
        self.device = self._get_device()
        # Embedder model to use
        self.model = "paraphrase-mpnet-base-v2"
        # Defining the embedder
        self.embedder = self._get_embedder()

        # Corpus embeddings
        self.source_colname = kwargs.get(
            "source_colname",
            "summary",
        )
        self.embeddings_colname = kwargs.get(
            "embeddings_colname",
            dv.embeddings_colname,
        )

        # Variables used for running semantic search
        self.corpus_dataset_with_faiss_index = kwargs.get(
            "corpus_dataset_with_faiss_index"
        )

    def _get_device(self) -> str:
        """
        Method for determining the device to use.

        Returns
        ----------
        device_type : str
            Type of device to use (e.g. 'cpu' or 'cuda').

            Options:
                - ``cpu``  : Uses a CPU.
                - ``cuda`` : Uses a GPU.
        """
        # Determining the type of device to use
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f">> Running on a '{device_type.upper()}' device")

        return device_type

    def _get_embedder(self):
        """
        Method for extracting the Embedder model.

        Returns
        ---------
        embedder : model
            Variable corresponding to the Embeddings models.
        """
        embedder = SentenceTransformer(self.model)
        embedder.to(self.device)

        return embedder

    def generate_corpus_index_and_embeddings(
        self,
        corpus_dataset: Dataset,
    ) -> Dataset:
        """
        Method for generating the Text Embeddings and FAISS indices from
        the input dataset.

        Parameters
        ------------
        corpus_dataset : datasets.Dataset
            Dataset containing the text to use to create the text
            embeddings and FAISS indices.

        Returns
        ----------
        corpus_dataset_with_embeddings : datasets.Dataset
            Dataset containing the original data rom ``corpus_dataset``
            plus the corresponding text embeddings of the ``source_colname``
            column.
        """
        torch.set_grad_enabled(False)

        # --- Generate text embeddings for the source column
        corpus_dataset_with_embeddings = corpus_dataset.map(
            lambda corpus: {
                self.embeddings_colname: self.embedder.encode(
                    corpus[self.source_colname]
                )
            },
            batched=True,
            desc="Computing Semantic Search Embeddings",
        )

        # --- Adding FAISS index
        corpus_dataset_with_embeddings.add_faiss_index(
            column=self.embeddings_colname,
            faiss_verbose=True,
            device=None if self.device == "cpu" else 1,
        )

        return corpus_dataset_with_embeddings

    def run_semantic_search(
        self,
        query: str,
        top_n: Optional[int] = 5,
    ) -> Dict:  # sourcery skip: extract-duplicate-method
        """
        Method for running a semantic search on a query after having
        created the corpus of the text embeddings.

        Parameters
        --------------
        query : str
            Text query to use for searching the database.

        top_n : int, optional
            Variable corresponding to the 'Top N' values to return based on the
            similarity score between the input query and the corpus. This
            variable is set to ``10`` by default.

        Returns
        ---------
        match_results : dict
            Dictionary containing the metadata of each of the articles
            that were in the Top-N in terms of being most similar to the
            input query ``query``.
        """
        # --- Checking input parameters
        # 'query' - Type
        query_type_arr = (str,)
        if not isinstance(query, query_type_arr):
            msg = ">> 'query' ({}) is not a valid input type ({})".format(
                type(query), query_type_arr
            )
            logger.error(msg)
            raise TypeError(msg)
        # 'top_n' - Type
        top_n_type_arr = (int,)
        if not isinstance(top_n, top_n_type_arr):
            msg = ">> 'top_n' ({}) is not a valid input type ({})".format(
                type(top_n), top_n_type_arr
            )
            logger.error(msg)
            raise TypeError(msg)

        # 'top_n' - Value
        if top_n <= 0:
            msg = f">> 'top_n' ({top_n}) must be larger than '0'!"
            logger.error(msg)
            raise ValueError(msg)

        # --- Checking that the encoder has been indexed correctly
        if self.corpus_dataset_with_faiss_index is None:
            msg = ">>> The FAISS index was not properly set!"
            logger.error(msg)
            raise ValueError(msg)

        # --- Encode the input query and extract the embedding
        query_embedding = self.embedder.encode(query)

        # --- Extracting the top-N results
        (
            scores,
            results,
        ) = self.corpus_dataset_with_faiss_index.get_nearest_examples(
            self.embeddings_colname,
            query_embedding,
            k=top_n,
        )

        # --- Sorting from highest to lowest
        # NOTE: We need to deconstruct the 'results' to be able to organize
        # the results
        parsed_results = pd.DataFrame.from_dict(
            data=results,
            orient="columns",
        )
        parsed_results.loc[:, "relevance"] = scores

        # Sorting in descending order
        parsed_results = parsed_results.sort_values(
            by=["relevance"],
            ascending=False,
        ).reset_index(drop=True)

        # Casting data type for the 'relevance'
        parsed_results.loc[:, "relevance"] = parsed_results["relevance"].apply(
            lambda x: str(np.round(x, 5))
        )

        # Only keeping certain columns
        columns_to_keep = ["_id", "title", "relevance", "content"]

        return parsed_results[columns_to_keep].to_dict(orient="index")
