
from typing import Union, Iterable

import numpy as np
from numpy import ndarray
from keras.datasets import mnist
from keras.utils import to_categorical
import os

def preprocess_func(local_file_path) -> Union[ndarray, Iterable, int, float, tuple, dict]:
    # Check if the data directory exists, and create it if not
    if local_file_path is None:
        home_dir = os.getenv("HOME")
        local_file_path = os.path.join(home_dir, "Tensorleap_data", 'mnist')

    if not os.path.exists(local_file_path):
        os.makedirs(local_file_path)

    data_file = os.path.join(local_file_path, 'mnist.npz')

    if not os.path.exists(data_file):
        # Data file doesn't exist, download and save it
        (train_X, train_Y), (val_X, val_Y) = mnist.load_data()
        train_X = np.expand_dims(train_X, axis=-1)  # Reshape :,28,28 -> :,28,28,1
        train_X = train_X / 255  # Normalize to [0,1]
        train_Y = to_categorical(train_Y)  # Hot Vector

        val_X = np.expand_dims(val_X, axis=-1)  # Reshape :,28,28 -> :,28,28,1
        val_X = val_X / 255  # Normalize to [0,1]
        val_Y = to_categorical(val_Y)  # Hot Vector

        # Save the data
        np.savez(data_file, train_X=train_X, val_X=val_X, train_Y=train_Y, val_Y=val_Y)

    data = np.load(data_file)
    return data





