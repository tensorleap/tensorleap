from typing import List, Union
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def preprocess(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    data_X = df.drop('label', axis=1).to_numpy()
    data_X = np.reshape(data_X, (len(data_X), 28, 28, 1)) / 255.  # normalize to range 0-1
    data_Y = df.label.to_numpy()
    # one hot encode the target values
    data_Y = to_categorical(data_Y)
    return data_X, data_Y


# load train and test mnist dataset localy
def load_datasets(train_file_path: Path, test_file_path: Path) -> List[np.ndarray]:

    df = pd.read_csv(train_file_path)
    train_X, train_Y = preprocess(df)

    df = pd.read_csv(test_file_path)
    test_X, test_Y = preprocess(df)

    # summarize loaded dataset
    print(f"Train: X={train_X.shape[0]}, y={train_Y.shape[0]}")
    print(f"Test: X={test_X.shape[0]}, y={test_Y.shape[0]}")

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


