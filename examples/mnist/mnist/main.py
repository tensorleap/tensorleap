import numpy as np
from pathlib import Path
from load_data import load_dataset, prep_pixels
from model import build_model, model_infer_one_sample

# example of loading the mnist dataset and building baseline cnn model


# run the test harness for evaluating a model
def run_test_harness() -> np.ndarray:
    train_file_path = Path("data", "mnist_train.csv").resolve()
    test_file_path = Path("data", "mnist_test.csv").resolve()
    # load dataset
    train_X, train_Y, test_X, test_Y = load_dataset(train_file_path, test_file_path)
    # prepare pixel data
    train_X, test_X = prep_pixels(train_X, test_X)
    # build the model
    model = build_model()
    # infer the model
    return model_infer_one_sample(train_X, model)

# Some sanity check
if __name__ == "__main__":
    # run the test harness
    y = run_test_harness()
    print(f"model raw output on one sample: {y}")
