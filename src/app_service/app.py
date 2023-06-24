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
from typing import Dict

import gradio as gr
from datasets import Dataset
from huggingface_hub import hf_hub_download

from src.classes import hugging_face_utils as hf
from src.classes import semantic_search_engine as ss
from src.utils import default_variables as dv

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
)
logger.setLevel(logging.INFO)


# ------------------------------ VARIABLES ------------------------------------

APP_TITLE = "Cicero LLM Synthesizer"
APP_DESCRIPTION = f"""
The '{APP_TITLE}'is an app that will identify the top-N articles from the
Cicero database that are most similar to the user's input query.
"""
APP_VERSION = "0.1"


# ------------------------------ FUNCTIONS ------------------------------------


def download_dataset_and_faiss_index() -> Dataset:
    """
    Function to download the corresponding dataset and the FAISS index
    from HuggingFace.

    Returns
    -------------
    dataset_with_faiss_index : datasets.Dataset
        Dataset from HuggingFace with the FAISS index loaded.
    """
    # --- Initializing HuggingFace API
    # Object for interacting with HuggingFace
    hf_obj = hf.HuggingFaceHelper()

    # Defining variable names for each of the objects
    faiss_index_name = f"{dv.faiss_index_name}.faiss"
    dataset_name = dv.dataset_faiss_embeddings_name
    username = hf_obj.username
    repository_name = dv.hugging_face_repository_name
    repository_id = f"{username}/{repository_name}"
    repository_type = "dataset"
    split_type = "train"

    # --- Downloading FAISS Index
    faiss_index_local_path = hf_hub_download(
        repo_id=repository_id,
        filename=faiss_index_name,
        repo_type=repository_type,
        token=hf_obj.api.token,
    )

    # --- Downloading Dataset
    dataset_obj = hf_obj.get_dataset_from_hub(
        dataset_name=dataset_name,
        username=username,
        split=split_type,
    )

    # --- Adding FAISS index to the dataset
    dataset_obj.load_faiss_index(
        index_name=dv.embeddings_colname,
        file=faiss_index_local_path,
    )

    return dataset_obj


def run_semantic_search_task(query: str, number_articles: int) -> Dict:
    # sourcery skip: remove-unnecessary-cast
    """
    Function to run semantic search on an input query. It will return a
    set of 'Top-N' articles that are most similar to the input query.

    Parameters
    ------------
    query : str
        Input query to use when running the Semantic Search Engine.

    number_articles : int
        Number of articles to return from the Semantic Search.

    Returns
    ----------
    ranked_results : dict
        Dictionary containing the ranked results from the Semantic
        Search Engine.
    """
    # --- Extracting dataset with FAISS index
    corpus_dataset_with_faiss_index = download_dataset_and_faiss_index()

    # --- Initializing Semantic Search Engine
    semantic_search_obj = ss.SemanticSearchEngine(
        corpus_dataset_with_faiss_index=corpus_dataset_with_faiss_index
    )

    # --- Running search on Top-N results
    number_articles_mod = int(number_articles)

    results = semantic_search_obj.run_semantic_search(
        query=query,
        top_n=number_articles_mod,
    )

    return list(results.values())


# --------------------------------- APP ---------------------------------------

# -- Semantic Search Engine
semantic_search_engine = gr.Interface(
    fn=run_semantic_search_task,
    inputs=[
        gr.components.Textbox(label="Input Query"),
        gr.Slider(
            minimum=1,
            label="Choose number of documents to retrieve",
            step=1,
        ),
    ],
    outputs="json",
    title=APP_TITLE,
    description=APP_DESCRIPTION,
)


# ----------------------------- RUNNING APP -----------------------------------

if __name__ == "__main__":
    semantic_search_engine.launch(
        debug=False,
        share=False,
        server_port=7860,
    )
