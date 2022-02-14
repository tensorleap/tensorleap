"""
This model is adapted from https://www.tensorflow.org/tutorials/keras/text_classification
The dataset used is IMDB taken from https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
as seen in the tensorflow example
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
from utils import standartize
from tensorflow.keras.preprocessing.sequence import pad_sequences
max_features = 10000
sequence_length = 250
embedding_dim = 16


def tensorleap_model():
    vectorized_inputs = tf.keras.Input(shape=250, dtype="int64")
    model = tf.keras.Sequential([
        layers.Embedding(max_features + 1, embedding_dim),
        layers.Dropout(0.2),
        layers.Dense(28, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(2, activation='sigmoid')])
    x = vectorized_inputs
    for lr in model.layers:
        x = lr(x)
    output = layers.Softmax(axis=-1)(x)
    tl_model = tf.keras.Model(inputs=vectorized_inputs, outputs=output)
    return tl_model


def infer_tensorleap_model(tokanizer_path: str) -> np.ndarray:
    model = tensorleap_model()
    text_input = "this is a test"
    with open(tokanizer_path, 'r') as f:
        data = json.load(f)
        tokanizer = tokenizer_from_json(data)
    standard_text = standartize(text_input)
    tokanized_input = tokanizer.texts_to_sequences([standard_text])
    padded_input = pad_sequences(tokanized_input, maxlen=sequence_length)
    return model.predict(padded_input)
