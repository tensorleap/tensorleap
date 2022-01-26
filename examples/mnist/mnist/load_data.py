from typing import List, Union
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


def read_csv_file(file_path: Path):
    df = pd.read_csv(file_path)
    data_X = df.drop('label', axis=1).to_numpy()
    data_X = np.reshape(data_X, (len(data_X), 28, 28, 1))
    data_Y = df.label.to_numpy()
    return data_X, data_Y


# load train and test mnist dataset
def load_dataset(train_file_path: Path, test_file_path: Path) -> List[np.ndarray]:

    # load datasets
    train_X, train_Y = read_csv_file(train_file_path)
    test_X, test_Y = read_csv_file(test_file_path)

    # (train_X, train_Y), (test_X, test_Y) = mnist.load_data()   # tensorflow dataset loading API

    # reshape dataset to have a single channel
    train_X = train_X.reshape((train_X.shape[0], 28, 28, 1))
    test_X = test_X.reshape((test_X.shape[0], 28, 28, 1))
    # summarize loaded dataset
    print(f"Train: X={train_X.shape[0]}, y={train_Y.shape[0]}")
    print(f"Test: X={test_X.shape[0]}, y={test_Y.shape[0]}")
    # one hot encode the target values
    train_Y = to_categorical(train_Y)
    test_Y = to_categorical(test_Y)
    return train_X, train_Y, test_X, test_Y


# scale pixels
def prep_pixels(train: np.ndarray, test: np.ndarray) -> List[np.ndarray]:
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm


