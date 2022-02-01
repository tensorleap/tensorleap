import numpy as np
import pandas as pd
from typing import Tuple
from pathlib import Path
from tensorflow.keras.utils import to_categorical
from model import build_model, model_infer_one_sample

# example of loading the mnist dataset and building baseline cnn model
def preprocess(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    data_X = df.drop('label', axis=1).to_numpy()
    data_X = np.reshape(data_X, (len(data_X), 28, 28, 1)) / 255.  # normalize to range 0-1
    data_Y = df.label.to_numpy()
    # one hot encode the target values
    data_Y = to_categorical(data_Y)
    return [data_X, data_Y]


# run the test harness for evaluating a model
def run_test_harness() -> np.ndarray:
    train_file_path = Path("data", "mnist_train.csv").resolve()
    test_file_path = Path("data", "mnist_test.csv").resolve()
    # load dataset
    df = pd.read_csv(train_file_path)
    train_X, train_Y = preprocess(df)
    print('check we normalize the data:', np.max(train_X))
    # build the model
    model = build_model()
    # infer the model
    return model_infer_one_sample(train_X, model)

# Some sanity check
if __name__ == "__main__":
    # run the test harness
    y = run_test_harness()
    print(f"model raw output on one sample: {y}")
