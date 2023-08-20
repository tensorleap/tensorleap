from typing import Tuple, Optional, Any, Union, Iterable

import numpy as np
from numpy import ndarray
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10
from keras.utils import to_categorical
import os


def preprocess_func(local_file_path) -> Union[ndarray, Iterable, int, float, tuple, dict]:
    # Check if the data directory exists, and create it if not
    if local_file_path is None:
        home_dir = os.getenv("HOME")
        local_file_path = os.path.join(home_dir, "Tensorleap_data", 'cifar10_resnet')

    if not os.path.exists(local_file_path):
        os.makedirs(local_file_path)

    data_file = os.path.join(local_file_path, 'cifar10_data.npz')

    if not os.path.exists(data_file):
        # Data file doesn't exist, download and save it
        (data_X, data_Y), (test_X, test_Y) = cifar10.load_data()
        data_Y = np.squeeze(data_Y)  # Normalize to [0, 1]
        data_Y = to_categorical(data_Y)  # One-hot encoding

        train_X, val_X, train_Y, val_Y = train_test_split(data_X, data_Y, test_size=0.2, random_state=42)

        # Save the data
        np.savez(data_file, train_X=train_X, val_X=val_X, train_Y=train_Y, val_Y=val_Y, test_X=test_X, test_Y=test_Y)

    data = np.load(data_file)
    return data
