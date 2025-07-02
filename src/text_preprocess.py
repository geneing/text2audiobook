from nemo_text_processing.text_normalization.normalize import Normalizer
from typing import List


class TextProcessor:
    """
    A class for text processing using NeMo's Normalizer.
    """

    def __init__(self, input_case: str = 'cased', lang: str = 'en'):
        """
        Initializes the TextProcessor with a Normalizer instance.

        Args:
            input_case (str): The input case for the normalizer. Defaults to 'cased'.
            lang (str): The language for the normalizer. Defaults to 'en'.
        """
        self.normalizer = Normalizer(input_case=input_case, lang=lang)

    def sentence_splitter(self, text: str) -> List[str]:
        """
        Splits the input text into sentences.

        Args:
            text (str): The text to split.

        Returns:
            A list of sentences.
        """
        # Using the normalizer's sentence splitting functionality
        return self.normalizer.split_text_into_sentences(text)

    def normalize(self, text: str | List[str], punct_post_process: bool = False, n_jobs: int = -2, batch_size: int = 100) -> str | List[str]:
        """
        Normalizes the input text.

        Args:
            text (str | List[str]): The input text to normalize.
            punct_post_process (bool): Whether to apply post-processing on punctuation.

        Returns:
            The normalized text.
        """
        # Using the normalizer's normalize function
        if isinstance(text, str):
            return self.normalizer.normalize(text, punct_post_process=punct_post_process)
        else:
            return self.normalizer.normalize_list(text, punct_post_process=punct_post_process, n_jobs=n_jobs, batch_size=batch_size)
