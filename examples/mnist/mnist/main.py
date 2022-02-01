import numpy as np
import pandas as pd
from typing import Tuple
from pathlib import Path
from tensorflow.keras.utils import to_categorical
from mnist.model import build_model, model_infer_one_sample

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



from code_loader.contract.datasetclasses import SubsetResponse
from mnist.main import preprocess
from mnist.model import build_model, model_infer_one_sample
from pathlib import Path
import pandas as pd
import numpy as np

def calc_classes_centroid(subset: SubsetResponse) -> dict:
    """ per each class we calculate average image on the pixels.
     returns dictionary key: class, values: images 28x28  """
    avg_images_dict = {}
    data_X = subset.data['images']
    data_Y = subset.data['labels']
    labels = np.arange(10).astype(str).tolist()
    for label in labels:
        inputs_label = data_X[np.equal(np.argmax(data_Y, axis=1), int(label))]
        avg_images_dict[label] = np.mean(inputs_label, axis=0)

    return avg_images_dict


def run_model_infer():
    train_file_path = Path("data", "mnist_train.csv").resolve()
    test_file_path = Path("data", "mnist_test.csv").resolve()
    # load dataset
    df = pd.read_csv(train_file_path)
    train_X, train_Y = preprocess(df)
    print('check we normalize the data:', np.max(train_X))
    # build the model
    model = build_model()

    # check for leap API
    train = SubsetResponse(length=200, data={'images': train_X,
                                                      'labels': train_Y
                                                      })
    avg_images_dict = calc_classes_centroid(train)
    return avg_images_dict





# Some sanity check
if __name__ == "__main__":
    # run the test harness
    y = run_test_harness()
    print(f"model raw output on one sample: {y}")
    avg_images_dict = run_model_infer()
    print(f"Done!")

