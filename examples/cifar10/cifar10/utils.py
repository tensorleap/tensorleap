import numpy as np
import yaml
from keras.datasets import cifar10
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

with open('/Users/chenrothschild/repo/tensorleap/examples/cifar10/project_config.yaml', 'r') as file:
    config_data = yaml.safe_load(file)

# Assign the constants to variables in the current file
LABELS_NAMES = config_data['LABELS_NAMES']

# Preprocess Function
def preprocess_func():
    """
    Description: Preprocesses the CIFAR-10 dataset by loading the data, normalizing the labels, splitting it into training and validation sets, and converting the labels to one-hot vectors.
    Returns:
    train_X (np.ndarray): Numpy array of shape (num_train_samples, image_height, image_width, num_channels) containing the training data.
    val_X (np.ndarray): Numpy array of shape (num_val_samples, image_height, image_width, num_channels) containing the validation data.
    train_Y (np.ndarray): Numpy array of shape (num_train_samples, num_classes) containing the one-hot encoded training labels.
    val_Y (np.ndarray): Numpy array of shape (num_val_samples, num_classes) containing the one-hot encoded validation labels.
    """
    (data_X, data_Y), _ = cifar10.load_data()

    # data_X = data_X / 255
    data_Y = np.squeeze(data_Y)  # Normalize to [0,1]
    data_Y = to_categorical(data_Y)  # Hot Vector

    train_X, val_X, train_Y, val_Y = train_test_split(data_X, data_Y, test_size=0.2, random_state=42)

    return train_X, val_X, train_Y, val_Y


# Metadata functions allow to add extra data for a later use in analysis.
# This metadata adds the int gt_digit of each sample (not a hot vector).
def metadata_gt_label(one_hot_digit: np.ndarray) -> int:
    """
     Retrieves the ground truth label (digit) of each sample from its one-hot encoded representation.
    :param:
    one_hot_digit (np.ndarray): One-hot encoded label of a single sample with shape (num_classes,).
    :return:
    digit_int (int): The ground truth label of the sample as an integer.
    """
    digit = one_hot_digit.argmax()  # Returns the indices of the maximum values along an axis
    digit_int = int(digit)
    return digit_int

# This metadata adds the int gt_name of each sample.
def metadata_label_name(one_hot_digit: np.ndarray) -> str:
    """
    Description: Retrieves the ground truth label name of each sample from its one-hot encoded representation.
    :param:
    one_hot_digit (np.ndarray): One-hot encoded label of a single sample with shape (num_classes,).
    :return:
    label_name (str): The ground truth label name of the sample as a string.
    """
    digit = one_hot_digit.argmax()  # Returns the indices of the maximum values along an axis
    digit_int = int(digit)
    return LABELS_NAMES[digit_int]


# This metadata adds each sample if it is contains to airplane or bird.
def metadata_fly(one_hot_digit: np.ndarray) -> str:
    """
    Description: Adds metadata to each sample indicating if it belongs to the "airplane or bird" category or not.
    :param:
    one_hot_digit (np.ndarray): One-hot encoded label of a single sample with shape (num_classes,).
    :return:
    metadata (str): Metadata indicating whether the sample belongs to the "airplane or bird" category or not.
    """
    digit = one_hot_digit.argmax()  # Returns the indices of the maximum values along an axis
    digit_int = int(digit)
    if digit_int in [0, 2]:
        return 'airplane or bird'
    return 'else'

# This metadata adds each sample if it is vehicle ar animal.
def metadata_animal(one_hot_digit: np.ndarray) -> str:
    """
    Description: Adds metadata to each sample indicating if it belongs to the "vehicles" category or the "animals" category.
    :param:
    one_hot_digit (np.ndarray): One-hot encoded label of a single sample with shape (num_classes,).
    :return:
    metadata (str): Metadata indicating whether the sample belongs to the "vehicles" category or the "animals" category.
    """
    digit = one_hot_digit.argmax()  # Returns the indices of the maximum values along an axis
    digit_int = int(digit)
    if digit_int in [0, 1, 9]:
        return 'vehicles'
    return 'animals'






