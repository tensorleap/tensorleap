import numpy as np
import re, string
from keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer

from IMDb.config import CONFIG


def standardize(comment: str) -> str:
    lowercase = comment.lower()
    html_stripped = re.sub('<br />', ' ', lowercase)
    punctuation_stripped = re.sub('[%s]' % re.escape(string.punctuation), '', html_stripped)
    return punctuation_stripped

def prepare_input(tokenizer, input_text: str) -> np.ndarray:
    standard_text = standardize(input_text)
    tokanized_input = tokenizer([standard_text], padding='max_length', truncation=True, max_length=CONFIG['SEQUENCE_LENGTH'])
    return tokanized_input
    # tokanized_input = tokenizer.texts_to_sequences([standard_text])
    # padded_input = pad_sequences(tokanized_input, maxlen=CONFIG['SEQUENCE_LENGTH'])
    # return padded_input[0, ...]