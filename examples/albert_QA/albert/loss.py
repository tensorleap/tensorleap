from typing import List
import tensorflow as tf
import numpy as np

def get_start_end_arrays(array: np.ndarray) -> List[int]:
    start_arr = array[..., 0]
    end_arr = array[..., 1]
    return start_arr, end_arr

def CE_loss(ground_truth: tf.Tensor, prediction: tf.Tensor) -> tf.Tensor:
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    alpha = 1.0
    start_pred, end_pred = get_start_end_arrays(prediction)
    start_gt, end_gt = get_start_end_arrays(ground_truth)
    combined_loss = loss(start_gt, start_pred) + alpha * loss(end_gt, end_pred)
    return combined_loss