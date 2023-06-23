import os
import json
import sys
from typing import List, Optional

import requests

import tiktoken


class Summarizer:
    def __init__(self, **kwargs):
        self.openai_endpoint = "https://api.openai.com/v1/chat/completions"

        # Prompt template
        self.prompt_template = self._get_prompt_template()

        # Type of model to use
        self.model = kwargs.get("model", "gpt-3.5-turbo")

        # Model hyperparameters
        self.max_tokens = kwargs.get("max_tokens", 4096)
        self.result_tokens = kwargs.get("result_tokens", 300)

        # Model encoding
        self.model_encoding = self._get_model_encoding()

        # Token length of the prompt template
        self.prompt_token_length = self._get_number_of_tokens(
            self.prompt_template
        )

    def _get_prompt_template(self) -> str:
        # Defining the template to use
        template_text = """
    Create a concise, clear, and in-depth summary of the following online article. Adhere to the following guidelines:

    1. Sound professional, detached and avoid emotionally charged language.
    2. Make sure to describe who is discussed in the article, what are
the events or concepts, when things happened, and, if this information is
available, why.
    3. The summary should be between one and three paragraphs.
    """

        return template_text

    def _get_model_encoding(self):
        return tiktoken.encoding_for_model(self.model)

    def _get_number_of_tokens(self, input_text: str) -> int:
        """
        Method for determining the number of tokens of the input text.

        Parameters
        -----------
        input_text : str
            Text to use for calculating its token length.

        Returns
        ---------
        text_token_length : int
            Lenght of the tokens of the input text.
        """

        return len(self.model_encoding.encode(input_text))

    def _run_model(
        self,
        user_content: str,
        temperature: Optional[float] = 1,
    ):
        """
        Method for running the model that will create the summary for a given
        observation.

        Parameters
        ------------
        user_content : str
            Content by the user that will be sent to the model via its API.

        temperature : float, optional
            Amount of ``temperature`` to give to the model. This parameter
            handles the amount of creativity that the model can have when
            creating the output response. This variable is set to ``1`` by
            default.

        Returns
        ----------
        """
        # Creating the headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f'Bearer {os.environ["OPENAI_API_KEY"]}',
        }
        # Composing the input messages
        messages = [
            {"role": "system", "content": self.prompt_template},
            {"role": "user", "content": user_content},
        ]
        # Parsing the request data
        request_data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        # Extracting the response from the model's API
        response = requests.post(
            self.openai_endpoint,
            headers=headers,
            data=json.dumps(request_data),
            timeout=60
        )

        # Checkig if the response was OK
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise RuntimeError(
                f"HTTP request failed code {response.status_code}, {response.text}"
            )

    def summarize(self, title, content):
        content_for_summary = f"{title}\n\n{content}"
        data_token_length = self._get_number_of_tokens(content_for_summary)
        while data_token_length + self.prompt_token_length > self.max_tokens - 10:
            print("Decimating the content.")
            content = content.split()
            del content[::10]
            content = ' '.join(content)
            content_for_summary = f"{title}\n\n{content}"
            data_token_length = self._get_number_of_tokens(content_for_summary)

        while True:
            try:
                return self._run_model(user_content=content_for_summary)
            except Exception as e:
                print(e, file=sys.stderr)
