import numpy as np
import re, string
from tensorflow.keras.preprocessing.sequence import pad_sequences
from IMDb.config import CONFIG


def standardize(comment: str) -> str:
    '''
    Standardizes a text comment by converting it to lowercase, stripping HTML tags, and removing punctuation.
    :param comment: The input text comment to be standardized.
    :return: The standardized text comment.
    '''
    lowercase = comment.lower()
    html_stripped = re.sub('<br />', ' ', lowercase)
    punctuation_stripped = re.sub('[%s]' % re.escape(string.punctuation), '', html_stripped)
    return punctuation_stripped

def prepare_input(tokenizer, input_text: str) -> np.ndarray:
    '''
    Prepares the input text by standardizing, tokenizing, and formatting it for model input.

    :param tokenizer: The tokenizer used for tokenization.
    :param input_text: The input text to be prepared.
    :return: A NumPy array containing the tokenized and formatted input.
    '''
    standard_text = standardize(input_text)
    tokanized_input = tokenizer([standard_text], padding='max_length', truncation=True, max_length=CONFIG['SEQUENCE_LENGTH'])
    return tokanized_input

def prepare_input_dense_model(tokenizer, input_text: str) -> np.ndarray:
    '''
    Prepares the input text by standardizing, tokenizing, and formatting it for model input.

    :param tokenizer: The tokenizer used for tokenization.
    :param input_text: The input text to be prepared.
    :return: A NumPy array containing the tokenized and formatted input.
    '''
    standard_text = standardize(input_text)
    tokanized_input = tokenizer.texts_to_sequences([standard_text])
    padded_input = pad_sequences(tokanized_input, maxlen=CONFIG['SEQUENCE_LENGTH'])
    return padded_input[0, ...]