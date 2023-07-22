import numpy as np

from keras.datasets import cifar10
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

LABELS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# Preprocess Function
def preprocess_func():
    (data_X, data_Y), _ = cifar10.load_data()

    # data_X = data_X / 255
    data_Y = np.squeeze(data_Y)  # Normalize to [0,1]
    data_Y = to_categorical(data_Y)  # Hot Vector

    train_X, val_X, train_Y, val_Y = train_test_split(data_X, data_Y, test_size=0.2, random_state=42)

    return train_X, val_X, train_Y, val_Y


# Metadata functions allow to add extra data for a later use in analysis.
# This metadata adds the int gt_digit of each sample (not a hot vector).
def metadata_gt_label(one_hot_digit: np.ndarray) -> int:
    digit = one_hot_digit.argmax()  # Returns the indices of the maximum values along an axis
    digit_int = int(digit)
    return digit_int

# This metadata adds the int gt_name of each sample.
def metadata_label_name(one_hot_digit: np.ndarray) -> str:
    digit = one_hot_digit.argmax()  # Returns the indices of the maximum values along an axis
    digit_int = int(digit)
    return LABELS_NAMES[digit_int]


# This metadata adds each sample if it is contains to airplane or bird.
def metadata_fly(one_hot_digit: np.ndarray) -> str:
    digit = one_hot_digit.argmax()  # Returns the indices of the maximum values along an axis
    digit_int = int(digit)
    if digit_int in [0, 2]:
        return 'airplane or bird'
    return 'else'



# This metadata adds each sample if it is vehicle ar animal.
def metadata_animal(one_hot_digit: np.ndarray) -> str:
    digit = one_hot_digit.argmax()  # Returns the indices of the maximum values along an axis
    digit_int = int(digit)
    if digit_int in [0, 1, 9]:
        return 'vehicles'
    return 'animals'






