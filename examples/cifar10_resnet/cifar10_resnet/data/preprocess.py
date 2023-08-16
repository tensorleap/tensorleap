import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10
from keras.utils import to_categorical


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
