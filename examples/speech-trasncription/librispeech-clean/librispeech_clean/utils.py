import numpy as np
from scipy.signal import find_peaks

from librispeech_clean.configuration import config
from librosa.feature import rms


def remove_trailing_zeros(original_array: np.ndarray) -> np.ndarray:
    """
    Remove trailing zeros from a NumPy array.

    This function takes a NumPy array as input and removes any trailing zeros from the end of the array. Trailing zeros
    are elements at the end of the array with a value of 0. The function returns a new array with trailing zeros removed.

    Args:
        original_array: A NumPy array from which trailing zeros should be removed.

    Returns:
        A new NumPy array with trailing zeros removed from the original_array. If the input array contains no trailing
        zeros, the function returns the original_array as is.

    Example:
        original_array = np.array([1, 2, 0, 0, 3, 0, 4])
        trimmed_array = remove_trailing_zeros(original_array)
        # trimmed_array will be np.array([1, 2, 0, 0, 3, 0, 4]) since there are trailing zeros.

        original_array = np.array([1, 2, 3, 4])
        trimmed_array = remove_trailing_zeros(original_array)
        # trimmed_array will be np.array([1, 2, 3, 4]) since there are no trailing zeros.
    """

    non_zero_indices = np.where(abs(original_array[::-1]) != 0)[0]

    if non_zero_indices.size > 0:
        last_non_zero_index = original_array.size - non_zero_indices[0] - 1
    else:
        last_non_zero_index = -1

    trimmed_array = original_array[:last_non_zero_index + 1]
    return trimmed_array


def pad_gt_numeric_labels(numeric_labels: np.ndarray) -> np.ndarray:
    padding_length = config.get_parameter('max_gt_length') - len(numeric_labels)
    return np.pad(numeric_labels, (0, padding_length))


def detect_pauses(signal: np.ndarray):
    data_trimmed = remove_trailing_zeros(signal)
    rms_res = normalize_array(rms(y=data_trimmed, frame_length=int(1e3)))
    local_minima = find_peaks(-rms_res[0], prominence=1e-1)[0]
    return local_minima


def normalize_array(data: np.ndarray, scale_factor: float = 1.0) -> np.ndarray:
    return scale_factor * (data - data.min()) / (data.max() - data.min())


def get_speech_pauses_to_word_gaps(audio_array: np.ndarray, reference: str) -> int:
    word_gaps = len(reference.split()) - 1
    pauses_indices = detect_pauses(audio_array)
    return len(pauses_indices) - word_gaps


def numeric_labels_to_text():
    pass


def logits_to_text():
    pass
