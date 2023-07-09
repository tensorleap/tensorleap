import numpy as np

from albert.data_set import get_start_position, max_sequence_length, get_end_position


def gt_index_encoder(sample, inputs) -> np.ndarray:
    start_position = get_start_position(sample, inputs)
    one_hot = np.zeros((max_sequence_length, 2))
    if start_position < max_sequence_length:
        one_hot[start_position, 0] = 1
    else:
        print("answer start position is out of max sequence length")
    end_position = get_end_position(sample, inputs)
    if end_position < max_sequence_length:
        one_hot[end_position, 1] = 1
    else:
        print("answer end position is out of max sequence length")
    return one_hot

def gt_end_index_encoder(sample, inputs) -> np.ndarray:
    end_position = get_end_position(sample, inputs)
    one_hot = np.zeros(max_sequence_length)
    if end_position < max_sequence_length:
        one_hot[end_position] = 1
    else:
        print("answer start position is out of max sequence length")
    return one_hot

def gt_start_index_encoder(sample, inputs) -> np.ndarray:
    start_position = get_start_position(sample, inputs)
    one_hot = np.zeros(max_sequence_length)
    if start_position < max_sequence_length:
        one_hot[start_position] = 1
    else:
        print("answer start position is out of max sequence length")
    return one_hot