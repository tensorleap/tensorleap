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
from imdb.utils import standartize, TransformerBlock, TokenAndPositionEmbedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
max_features = 10000
sequence_length = 250
AUTOTUNE = tf.data.AUTOTUNE
embedding_dim = 16


def tensorleap_model():
    vectorized_inputs = tf.keras.Input(shape=sequence_length, dtype="int64")
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


def tensorleap_model_with_attention():
    vectorized_inputs = tf.keras.Input(shape=sequence_length, dtype="int64")
    positions = tf.keras.Input(shape=sequence_length, dtype="int64")
    attention_mask = tf.keras.Input(shape=(sequence_length, sequence_length), dtype="int64")
    transformer = TransformerBlock(embed_dim=embedding_dim,
                                   num_heads=2,
                                   ff_dim=28
                                   )
    embed = TokenAndPositionEmbedding(maxlen=sequence_length,
                              vocab_size=max_features+1,
                              embed_dim=embedding_dim)
    model = tf.keras.Sequential([
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(2, activation='sigmoid')])
    embedding = embed.call(vectorized_inputs, positions)
    transformer_output = transformer.call(embedding, attention_mask=attention_mask)
    x = transformer_output
    for lr in model.layers:
        x = lr(x)
    output = layers.Softmax(axis=-1)(x)
    tl_model = tf.keras.Model(inputs=[vectorized_inputs, positions, attention_mask], outputs=output)
    return tl_model