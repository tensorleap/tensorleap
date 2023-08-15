import numpy as np
from code_loader.contract.datasetclasses import PreprocessResponse

from cifar10_resnet.config import CONFIG


#

# Metadata functions allow to add extra data for a later use in analysis.
# This metadata adds the int gt_digit of each sample (not a hot vector).
def metadata_gt_label(digit_int: int) -> int:
    """
     Retrieves the ground truth label (digit) of each sample from its one-hot encoded representation.
    :param:
    one_hot_digit (np.ndarray): One-hot encoded label of a single sample with shape (num_classes,).
    :return:
    digit_int (int): The ground truth label of the sample as an integer.
    """
    return digit_int

# This metadata adds the int gt_name of each sample.
def metadata_label_name(digit_int: int) -> str:
    """
    Description: Retrieves the ground truth label name of each sample from its one-hot encoded representation.
    :param:
    one_hot_digit (np.ndarray): One-hot encoded label of a single sample with shape (num_classes,).
    :return:
    label_name (str): The ground truth label name of the sample as a string.
    """
    return CONFIG['LABELS_NAMES'][digit_int]


# This metadata adds each sample if it is contains to airplane or bird.
def metadata_fly(digit_int: int) -> str:
    """
    Description: Adds metadata to each sample indicating if it belongs to the "airplane or bird" category or not.
    :param:
    one_hot_digit (np.ndarray): One-hot encoded label of a single sample with shape (num_classes,).
    :return:
    metadata (str): Metadata indicating whether the sample belongs to the "airplane or bird" category or not.
    """
    if digit_int in [0, 2]:
        return 'airplane or bird'
    return 'else'

# This metadata adds each sample if it is vehicle ar animal.
def metadata_animal(digit_int: int) -> str:
    """
    Description: Adds metadata to each sample indicating if it belongs to the "vehicles" category or the "animals" category.
    :param:
    one_hot_digit (np.ndarray): One-hot encoded label of a single sample with shape (num_classes,).
    :return:
    metadata (str): Metadata indicating whether the sample belongs to the "vehicles" category or the "animals" category.
    """
    if digit_int in [0, 1, 9]:
        return 'vehicles'
    return 'animals'







