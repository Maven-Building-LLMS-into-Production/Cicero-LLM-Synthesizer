import itertools
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import packaging
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image as PilImage
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ABSAnalyzer(object):
    def __init__(self):
        """
        Class definition for the Aspect-based Sentiment Analysis (ABSA)
        service.

        Example:
        ---------
        >>> absa_obj = ABSAnalyzer()
        # Prepare the data for the ABSA service
        # dataset is a pandas.DataFrame containing the summaries of each
        # article
        # along with their ids and titles.
        >>> absa_obj.prepare_data(dataset)
        # Use article ids (list of strings) to get the aspect-based sentiment
        # scores for each article. User can also specify the aspects to be
        # considered. If no aspects are specified, the default ones will be
        # used.
        >>> absa_output = absa_service.get_scores_for_articles(ids, aspects)
        """
        # Defining data directory
        self.datasets_dir = Path().resolve().joinpath("datasets")

        # Defining class attributes
        self.model_name = "yangheng/deberta-v3-base-absa-v1.1"
        self.source_colname = "summary"

        # Tokenizer and model
        self.model = self._get_model()
        self.tokenizer = self._get_tokenizer()

        # Data preparation
        self.batch_size = 64
        self.__default_aspects = "refreshing, historical, innovative"
        self.dataset = None
        self.article_titles = None

    def _get_tokenizer(self):
        """
        Method for extracting the tokenizer of the application.

        Returns
        ---------
        tokenizer_obj :
            Tokenizer of the service.
        """

        return AutoTokenizer.from_pretrained(self.model_name)

    def _get_model(self):
        """
        Method for extracting the ABSA model of the application.

        Returns
        ---------
        model :
            ABSA model object.
        """
        # Defining the model
        model_obj = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )

        # Setting the model to evaluation mode
        model_obj.eval()

        # Compiling the model for faster inference, if applicable
        torch_version = packaging.version.parse(torch.__version__)
        if torch_version >= packaging.version.parse("2.0.0"):
            try:
                # Trying compilation
                model_obj = torch.compile(
                    model_obj, mode=(mode := "reduce-overhead")
                )
                logging.info(f">> Model compiled in {mode} mode!")
            except Exception as e:
                logging.warning(
                    f">> Model could not be compiled in {mode} mode!"
                )
                logging.warning(f">> Exception: {e}")

        return model_obj

    def prepare_data(self, dataset: pd.DataFrame):
        """
        Method for executing the data preparation tasks, such as defining
        the batches and the inputs to the model.

        Parameters
        -----------
        dataset : pandas.DataFrame
            Dataset containing the summaries of each article.
        """
        # Dropping duplicates
        dataset.drop_duplicates("_id", inplace=True)

        # Extracting the text for ABSA
        summaries = dataset[self.source_colname].tolist()

        # Extracting the article titles
        article_titles = dataset["title"].tolist()

        # Extracting ids
        ids = dataset["_id"].tolist()

        # --- Updating attributes
        self.dataset = dataset.copy()
        self.summaries = summaries
        self.article_titles = article_titles
        self.ids = ids
        self.id2summary = dict(zip(ids, summaries))
        self.id2title = dict(zip(ids, article_titles))

    def run(
        self,
        model_inputs: List[torch.Tensor],
        n_summaries: int,
        aspects: List[str],
    ) -> pd.DataFrame:
        """
        Method for running the 'Aspect-Based Sentiment Analysis' for the
        observations from the input dataset.

        Parameters
        ------------
        save_to_disk : bool, optional
            If ``True``, the output dataset will be saved to disk. This
            variable is set to ``True`` by default.

        Returns
        -----------
        pos_sentiment_score : pandas.DataFrame
            Dataset containing the sentiment score for each aspect of each
            article.
        """
        logging.info(">> Running ABSA generation ...")

        # Calculating the scores for each set of batches
        output_tensors = []

        start_time = datetime.now()
        logging.info(f">>   Start Time: {start_time}")

        for batch_id, batch in enumerate(model_inputs):
            with torch.no_grad():
                # Computing the inference
                batch_output = self.model(**batch)
                logits = batch_output["logits"]
                scores = F.softmax(logits, dim=1)
                output_tensors.append(scores)

                logging.info(
                    f">>    Batch: {batch_id} finished at: {datetime.now()}"
                )

        # Concatenating the scores of all batches
        scores_all_batches = torch.cat(output_tensors)

        end_time = datetime.now()
        logging.info(f">>   End time: {end_time}")
        logging.info(f">>   Took: {end_time - start_time}")

        # --- Reshaping the scores of all batches
        # NOTE: (N x num_aspects) x num_classes ->
        # N x num_aspects x num_classes
        scores_all_batches = scores_all_batches.reshape(
            n_summaries, len(aspects), -1
        )
        scores_all_batches = scores_all_batches.detach().numpy()

        # Extracting the labels for each 'aspect'
        label2id = {v: k for k, v in self.model.config.id2label.items()}
        positive_idx = label2id["Positive"]

        # --- Data parsing
        # Creating a DataFrame with the sentiment score for each aspect
        positive_sentiment_score = pd.DataFrame(
            scores_all_batches[..., positive_idx], columns=aspects
        )

        return positive_sentiment_score

    def _create_article_spider_chart(
        self,
        article_title: str,
        score: pd.DataFrame,
    ) -> PilImage:
        """
        Method for creating the Spider Chart for the article's aspects.

        Parameters
        ------------
        article_title : str
            Title of the article.

        score : pandas.DataFrame
            Dataset containing the data about the score of each aspect.
            Columns are the aspects, and row is the score for this article.

        Returns
        ----------
        article_spider_chart : PIL.Image
            Object corresponding to the Spider Chart of the article's aspects.
        """
        # Defining categories of the data
        categories = [c.capitalize() for c in score.index]

        # Calculating angles
        N = len(categories)
        values = values = score.values.tolist() + score.values.tolist()[:1]
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]

        # Creating figure
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(
            angles, values, linewidth=1, linestyle="solid", color="#F8766D"
        )
        ax.fill(angles, values, "#F8766D", alpha=0.5)
        ax.set_thetagrids(
            np.degrees(angles[:-1]),
            labels=categories,
            fontsize=12,
        )
        ax.set_title(article_title, fontsize=14)
        ax.grid(True)

        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        plt.ylim(0, 1)

        plt.tight_layout()

        # Temporary directory
        path_file = str(
            Path(self.datasets_dir).joinpath(f"{article_title}.jpg")
        )
        Path(path_file).parent.mkdir(exist_ok=True, parents=True)

        # Temporariy saving image
        plt.savefig(path_file)

        # Reading in image using Pillow
        article_spider_chart = PilImage.open(path_file)

        # Removing image
        Path(path_file).unlink()

        return article_spider_chart

    def get_scores_for_articles(
        self,
        article_ids: List[str],
        aspects: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, PilImage.Image]:
        """
        Method for extracting individual scores of an article, as well as the
        corresponding image.

        Parameters
        -------------
        article_ids : List[str]
            List containing the ids of the articles to extract the scores from.

        aspects : str
            String containing the aspects to extract the scores from.
            Aspects should be separated by commas if multiple aspects are
            specified.

        Returns
        -----------
        article_scores : dict
            Dictionary containing the scores for each aspect of the articles.

        article_spider_chart : PIL.Image
            Image corresponding to the Spider chart of the article.
        """
        # Checking that the data is available
        if self.dataset is None:
            msg = ">> You must first run 'prepare_data'!"
            logging.error(msg)
            raise ValueError(msg)

        summaries = []
        for _id in article_ids:
            if _id in self.id2summary:
                summaries.append(self.id2summary[_id])
            else:
                msg = f"Article with id: {_id} not found."
                raise ValueError(msg)
        n_summaries = len(summaries)

        # NOTE: Creating a list that contains copies of each summary for every
        # aspect, so that one can pass these to the model.

        if aspects is None:
            aspects = self.__default_aspects
        aspects = aspects.strip().split(",")
        assert len(aspects) > 0, "No aspects specified."

        n_aspects = len(aspects)
        repeated_summaries = list(
            itertools.chain.from_iterable(
                itertools.repeat(s, n_aspects) for s in summaries
            )
        )

        # Number of repeated 'aspects'
        repeated_aspects = aspects * n_summaries

        # Creating the batches of data to pass to the model
        batches = [
            (
                repeated_summaries[i : i + self.batch_size],
                repeated_aspects[i : i + self.batch_size],
            )
            for i in range(0, len(repeated_summaries), self.batch_size)
        ]

        # Set of inputs to the model
        model_inputs = [
            self.tokenizer(*batch, padding=True, return_tensors="pt")
            for batch in batches
        ]

        # Predicting the scores
        positive_sentiment_score = self.run(model_inputs, n_summaries, aspects)
        positive_sentiment_score["_id"] = article_ids
        positive_sentiment_score.set_index("_id", inplace=True)

        # Creating Spider-charts for each article
        article_spider_charts = []
        for _id, score in positive_sentiment_score.iterrows():
            article_title = self.id2title[_id]
            article_spider_chart = self._create_article_spider_chart(
                article_title=article_title,
                score=score,
            )
            article_spider_charts.append(article_spider_chart)

        return positive_sentiment_score, article_spider_charts
