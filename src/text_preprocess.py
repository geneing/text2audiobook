import os
import logging
import re

from nemo_text_processing.text_normalization.normalize import Normalizer
import pysbd

logger = logging.getLogger(__name__)


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
        self.segmenter = pysbd.Segmenter(language=lang, clean=False)

    def normalize(self, text: str | list[str], punct_post_process: bool = False, n_jobs: int = -2, batch_size: int = 100) -> str | list[str]:
        """
        Normalizes the input text.

        Args:
            text (str | list[str]): The input text to normalize.
            punct_post_process (bool): Whether to apply post-processing on punctuation.

        Returns:
            The normalized text.
        """
        # Using the normalizer's normalize function
        if isinstance(text, str):
            return self.normalizer.normalize(text, punct_post_process=punct_post_process)
        else:
            return self.normalizer.normalize_list(text, punct_post_process=punct_post_process, n_jobs=n_jobs, batch_size=batch_size)

    def sentence_splitter(self, text: str, max_chars: int = 1000) -> list[str]:
        """
        Split text into chunks, where each chunk is as close to max_chars as possible.
        
        Args:
            text: The text to split
            max_chars: The maximum number of characters per chunk
        
        Returns:
            A list of text chunks
        """
        # Edge cases
        if not text:
            return []
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        
        # Use sentencex to get all sentences
        sentences = list(self.segmenter.segment(text))
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Add a space between sentences if current_chunk is not empty
            space = " " if current_chunk else ""
            # Check if adding the sentence would exceed max_chars
            if len(current_chunk) + len(space) + len(sentence) <= max_chars:
                current_chunk += space + sentence
            # If the sentence itself is longer than max_chars, split it
            elif len(sentence) > max_chars:
                # Add the current chunk if it's not empty
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # Split the long sentence
                sentence_chunks = self.split_long_sentence(sentence, max_chars)
                
                # Add all chunks except the last one
                chunks.extend(sentence_chunks[:-1])
                
                # Start a new chunk with the last sentence chunk
                current_chunk = sentence_chunks[-1]
            # Otherwise, start a new chunk with this sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        # For DEBUG only
        # # Assert that no chunk exceeds max_chars
        # for i, chunk in enumerate(chunks):
        #     assert len(chunk) <= max_chars, f"Chunk {i} length {len(chunk)} exceeds max_chars {max_chars}"
        
        # # Assert that no content is lost (loose check)
        # original_sans_whitespace = ''.join(c for c in text if not c.isspace())
        # chunks_sans_whitespace = ''.join(c for c in ''.join(chunks) if not c.isspace())
        
        # # The lengths should be the same
        # assert len(chunks_sans_whitespace) == len(original_sans_whitespace), "Content might be lost during splitting"
        
        return chunks

    def paragraph_splitter(self, text: str) -> list[str]:
        """
        Splits text into paragraphs based on one or more empty lines.

        Args:
            text (str): The input text to split.

        Returns:
            list[str]: A list of paragraphs.
        """
        # Use regex to split by one or more empty lines, then clean up whitespace
        paragraphs = re.split(r'\n\s*\n', text)
        # Filter out any empty strings that may result from the split
        return [p.strip() for p in paragraphs if p.strip()]

    def split_long_sentence(self, sentence: str, max_chars: int) -> list[str]:
        """
        Split a long sentence into smaller parts based on punctuation and spaces.
        
        Args:
            sentence: The sentence to split
            max_chars: The maximum number of characters per part
        
        Returns:
            A list of sentence parts
        """
        # If max_chars is extremely small, split by character
        if max_chars < 5:
            return [sentence[i:i+max_chars] for i in range(0, len(sentence), max_chars)]
        
        # Define punctuation marks in order of priority
        punctuations = [
            '。', '！', '？',  # Chinese end-of-sentence
            '. ', '! ', '? ',  # English end-of-sentence with space
            '；', ';',  # Semicolons
            '，', ',',  # Commas
            '：', ':',  # Colons
            '）', ')', ']', '】', '}', '」', '』',  # Closing parentheses and brackets
            '、',  # Chinese enumeration comma
            '—', '-', '–',  # Dashes
            ' ',  # Spaces as last resort
        ]
        
        parts = []
        remaining = sentence
        
        while remaining:
            if len(remaining) <= max_chars:
                parts.append(remaining)
                break
            
            # Try to find the best split point based on punctuation marks
            best_split_idx = -1
            
            for punctuation in punctuations:
                # Find the rightmost occurrence of the punctuation within max_chars
                split_idx = remaining[:max_chars].rfind(punctuation)
                
                if split_idx != -1:
                    # For punctuation marks that are not spaces, include them in the current chunk
                    if punctuation != ' ':
                        split_idx += len(punctuation)
                    else:
                        # For spaces, exclude them from both chunks
                        split_idx += 1
                    
                    best_split_idx = split_idx
                    break
            
            # If no punctuation is found, split at max_chars
            if best_split_idx == -1:
                best_split_idx = max_chars
            
            parts.append(remaining[:best_split_idx])
            remaining = remaining[best_split_idx:]
        
        return parts
