import numpy as np

from squad_albert.config import CONFIG
from squad_albert.utils.utils import get_start_position, get_end_position


def gt_index_encoder(sample: dict, inputs: dict) -> np.ndarray:
    """
    Description: Encodes the ground truth start and end indices as a one-hot array for the given sample.
    Parameters:
    sample (dict): A dictionary containing the sample data, including the answer.
    inputs (dict): A dictionary containing inputs related to the sample, such as offset_mapping and token_type_ids.
    Returns:
    one_hot (np.ndarray): A one-hot array of shape (max_sequence_length, 2) with the ground truth start and end indices
    encoded. The first column represents the start index, and the second column represents the end index.
    """
    start_position = get_start_position(sample, inputs)
    one_hot = np.zeros((CONFIG['max_sequence_length'], 2))
    if start_position < CONFIG['max_sequence_length']:
        one_hot[start_position, 0] = 1
    else:
        print("answer start position is out of max sequence length")
    end_position = get_end_position(sample, inputs)
    if end_position < CONFIG['max_sequence_length']:
        one_hot[end_position, 1] = 1
    else:
        print("answer end position is out of max sequence length")
    return one_hot

def gt_end_index_encoder(sample: dict, inputs: dict) -> np.ndarray:
    """
    Description: Encodes the ground truth end index as a one-hot array for the given sample.
    Parameters:
    sample (dict): A dictionary containing the sample data, including the answer.
    inputs (dict): A dictionary containing inputs related to the sample, such as offset_mapping and token_type_ids.
    Returns:
    one_hot (np.ndarray): A one-hot array of shape (max_sequence_length,) with the ground truth end index encoded.
    """
    end_position = get_end_position(sample, inputs)
    one_hot = np.zeros(CONFIG['max_sequence_length'])
    if end_position < CONFIG['max_sequence_length']:
        one_hot[end_position] = 1
    else:
        print("answer start position is out of max sequence length")
    return one_hot

def gt_start_index_encoder(sample: dict, inputs: dict) -> np.ndarray:
    """
    Description: Encodes the ground truth start index as a one-hot array for the given sample.
    Parameters:
    sample (dict): A dictionary containing the sample data, including the answer.
    inputs (dict): A dictionary containing inputs related to the sample, such as offset_mapping and token_type_ids.
    Returns:
    one_hot (np.ndarray): A one-hot array of shape (max_sequence_length,) with the ground truth start index encoded.
    """
    start_position = get_start_position(sample, inputs)
    one_hot = np.zeros(CONFIG['max_sequence_length'])
    if start_position < CONFIG['max_sequence_length']:
        one_hot[start_position] = 1
    else:
        print("answer start position is out of max sequence length")
    return one_hot