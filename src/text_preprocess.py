from nemo_text_processing.text_normalization.normalize import Normalizer

normalizer = Normalizer(input_case='cased', lang='en')

def sentence_splitter(text):
    """
    Splits the input text into sentences.
    """
    # Using the normalizer's sentence splitting functionality
    return normalizer.split_text_into_sentences(text)

def normalize(text, punct_post_process=False) -> str | list[_T]:
    """
    Normalizes the input text.
    
    Args:
        text (str): The input text to normalize.
        punct_post_process (bool): Whether to apply post-processing on punctuation.
        
    Returns:
        str: The normalized text.
    """
    # Using the normalizer's normalize function
    if isinstance(text, str):       
        return normalizer.normalize(text, punct_post_process=punct_post_process)
    else:
        return normalizer.normalize_list(text, punct_post_process=punct_post_process, n_jobs=-2, batch_size=100)

