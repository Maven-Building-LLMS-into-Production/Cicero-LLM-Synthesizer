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
from pathlib import Path
from typing import Dict

from src.classes import hugging_face_utils as hf
from src.classes import semantic_search_engine as ss
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
        "--dataset-name",
        dest="dataset_name",
        default=dv.summaries_dataset_name,
        type=str,
        help="""
        Name of the HuggingFace dataset
        [Default: '%(default)s']
        """,
    )
    # Name of the output Dataset with FAISS index and embeddings
    parser.add_argument(
        "--output-dataset-name",
        dest="output_dataset_name",
        default=dv.dataset_faiss_embeddings_name,
        type=str,
        help="""
        Name of the output dataset that will contain a FAISS index the
        text embeddings of the summaries.
        [Default: '%(default)s']
        """,
    )
    # Name of the HuggingFace repository
    parser.add_argument(
        "--repository-name",
        dest="repository_name",
        default=dv.hugging_face_repository_name,
        type=str,
        help="""
        Name of the HuggingFace repository to use for storing artifacts.
        [Default: '%(default)s']
        """,
    )
    # Name of the FAISS Index
    parser.add_argument(
        "--faiss-index-name",
        dest="faiss_index_name",
        default=dv.faiss_index_name,
        type=str,
        help="""
        Name of the FAISS Index of the output dataset.
        [Default: '%(default)s']
        """,
    )

    return parser.parse_args()


# -------------------------------  FUNCTIONS ----------------------------------


def create_faiss_index_and_embeddings_from_dataset(params_dict: Dict):
    """
    Function to create a Dataset object with a FAISS index and the
    corresponding text embeddings.

    Parameters
    -----------
    params_dict : dict
        Dictionary with set of parameters that are used throughout the project.
    """
    # --- Initializing object for interacting with Datasets
    hf_obj = hf.HuggingFaceHelper()

    # --- Download dataset from HuggingFace Hub
    dataset_obj = hf_obj.get_dataset_from_hub(
        dataset_name=params_dict["dataset_name"],
        username=hf_obj.username,
        split="train",
    )

    # --- Generate the FAISS index and Text embeddings
    # Initialize Semantic Search engine
    semantic_search_obj = ss.SemanticSearchEngine()

    # Create FAISS index and the dataset with text embeddings
    dataset_with_embeddings_obj = (
        semantic_search_obj.generate_corpus_index_and_embeddings(
            corpus_dataset=dataset_obj
        )
    )

    # --- Extract FAISS index and upload it to HuggingsFace Hub
    # Path to the output file that will contain the FAISS index
    faiss_index_local_path = str(
        gu.get_project_paths()["data"].joinpath(
            f'{params_dict["faiss_index_name"]}.faiss'
        )
    )

    dataset_with_embeddings_obj.save_faiss_index(
        index_name=semantic_search_obj.embeddings_colname,
        file=faiss_index_local_path,
    )

    # Creating repository in HuggingFace
    repo_name = f'{hf_obj.username}/{params_dict["repository_name"]}'
    repo_type = "dataset"

    _ = hf_obj.api.create_repo(
        repo_id=repo_name,
        repo_type=repo_type,
        exist_ok=True,
    )

    # Uploading FAISS
    hf_obj.api.upload_file(
        path_or_fileobj=faiss_index_local_path,
        path_in_repo=Path(faiss_index_local_path).name,
        repo_id=repo_name,
        repo_type=repo_type,
    )

    # --- Upload new Dataset to HuggingFace
    # Dropping FAISS index
    dataset_with_embeddings_obj.drop_index(
        index_name=semantic_search_obj.embeddings_colname
    )

    # Pushing dataset to HuggingFace
    hf_obj.push_dataset(
        dataset=dataset_with_embeddings_obj,
        dataset_name=params_dict["output_dataset_name"],
        username=hf_obj.username,
    )

    return


# ------------------------------ MAIN FUNCTIONS -------------------------------


def main(params_dict: Dict):
    """
    Main function for creating a dataset with FAISS index.
    """
    # Showing set of input parameters
    gu.show_params(params_dict=params_dict, logger=logger)

    # Create FAISS index and Text embeddings for the dataset.
    create_faiss_index_and_embeddings_from_dataset(params_dict=params_dict)

    return


if __name__ == "__main__":
    # Getting input parameters
    params_dict = vars(get_parser())
    # Running main function
    main(params_dict=params_dict)
