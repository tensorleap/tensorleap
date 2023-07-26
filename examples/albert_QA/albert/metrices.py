from typing import List
import tensorflow as tf

def get_start_end_arrays(array) -> List[int]:
    """
    Description: Extracts the start and end elements from the last axis of the input array.
    Parameters:
    array (tf.Tensor): Input tensor of shape [B, max_sequence_length, 2].
    Returns:
    start_arr (tf.Tensor): Tensor containing the start elements of the input array with shape [B, max_sequence_length].
    end_arr (tf.Tensor): Tensor containing the end elements of the input array with shape [B, max_sequence_length].
    """
    start_arr = array[..., 0]
    end_arr = array[..., 1]
    return start_arr, end_arr

def exact_match_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:  # return batch
    """
    Description: checks if the prediction is identical for the gt and prediction
    Parameters:
    y_true :[B, max_sequence_length, 2]
    y_pred: [B, max_sequence_length, 2]
    Returns:
    exact_match (tf.Tensor): Binary tensor of shape [B] where 1 represents an exact match between predictions and ground truth for each sequence, and 0 represents a mismatch.
    """
    is_argmax_equal = tf.equal(tf.argmax(y_true, axis=1), tf.argmax(y_pred, axis=1))
    is_exact_match = tf.reduce_all(is_argmax_equal, axis=1)
    return tf.cast(is_exact_match, tf.float64)


def get_nonnegative_tensor(tensor: tf.Tensor) -> tf.Tensor:
    """
    Description: Replaces negative elements in the input tensor with zeros.
    Parameters:
    tensor (tf.Tensor): Input tensor.
    Returns:
    non_negative_tensor (tf.Tensor): Tensor with negative elements replaced by zeros.
    """
    return tf.where(tensor > 0, tensor, tf.zeros_like(tensor))


def f1_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:  # return batch
    """
    Description: Computes the F1 metric (2*precision*recall)/(precision + recall)
    Parameters:
    y_true (tf.Tensor): Ground truth tensor of shape [B, max_sequence_length, 2].
    y_pred (tf.Tensor): Predicted tensor of shape [B, max_sequence_length, 2].
    Returns:
    f1 (tf.Tensor): F1 scores for each sequence in the batch, with shape [B].
    """
    start_pred, end_pred = tf.transpose(tf.argmax(y_pred, axis=1))
    end_pred += 1  # fix end bondary encoding
    start_true, end_true = tf.transpose(tf.argmax(y_true, axis=1))
    end_true += 1  # fix end bondary encoding
    number_of_matches = tf.minimum(end_pred, end_true) - tf.maximum(start_pred, start_true)
    number_of_matches = get_nonnegative_tensor(number_of_matches)
    amount_predicted = get_nonnegative_tensor(end_pred - start_pred)
    amount_truth = get_nonnegative_tensor(end_true - start_true)
    precision = tf.where(amount_predicted > 0, number_of_matches / amount_predicted,
                         tf.zeros_like(amount_predicted, dtype=tf.float64))
    recall = tf.where(amount_truth > 0, number_of_matches / amount_truth, tf.zeros_like(amount_truth, dtype=tf.float64))
    f1 = tf.where(precision + recall > 0, 2 * precision * recall / (precision + recall),
                  tf.zeros_like(recall, dtype=tf.float64))
    return f1

def CE_start_index(ground_truth: tf.Tensor, prediction: tf.Tensor) -> tf.Tensor:
    """
    Description: Computes the Categorical Cross-Entropy loss for the start index predictions.
    Parameters:
    ground_truth (tf.Tensor): Ground truth tensor of shape [B, max_sequence_length, 2].
    prediction (tf.Tensor): Predicted tensor of shape [B, max_sequence_length, 2].
    Returns:
    loss (tf.Tensor): Categorical Cross-Entropy loss for the start index predictions.
    """
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    start_gt, end_gt = get_start_end_arrays(ground_truth)
    start_pred, end_pred = get_start_end_arrays(prediction)
    return loss(start_gt, start_pred)

def CE_end_index(ground_truth: tf.Tensor, prediction: tf.Tensor) -> tf.Tensor:
    """
    Description: Computes the Categorical Cross-Entropy loss for the end index predictions.
    Parameters:
    ground_truth (tf.Tensor): Ground truth tensor of shape [B, max_sequence_length, 2].
    prediction (tf.Tensor): Predicted tensor of shape [B, max_sequence_length, 2].
    Returns:
    loss (tf.Tensor): Categorical Cross-Entropy loss for the end index predictions.
    """
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    start_gt, end_gt = get_start_end_arrays(ground_truth)
    start_pred, end_pred = get_start_end_arrays(prediction)
    return loss(end_gt, end_pred)