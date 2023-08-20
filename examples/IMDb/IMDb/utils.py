import numpy as np

import re, string
from tensorflow.keras.preprocessing.sequence import pad_sequences

def standardize(comment: str) -> str:
    lowercase = comment.lower()
    html_stripped = re.sub('<br />', ' ', lowercase)
    punctuation_stripped = re.sub('[%s]' % re.escape(string.punctuation), '', html_stripped)
    return punctuation_stripped


def prepare_input(tokanizer, input_text: str, sequence_length: int = 250) -> np.ndarray:
    standard_text = standardize(input_text)
    tokanized_input = tokanizer.texts_to_sequences([standard_text])
    padded_input = pad_sequences(tokanized_input, maxlen=sequence_length)
    return padded_input[0, ...]