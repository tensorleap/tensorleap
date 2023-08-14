import tensorflow as tf

from squad_albert.metrics import get_start_end_arrays


def CE_loss(ground_truth: tf.Tensor, prediction: tf.Tensor) -> tf.Tensor:
    """
    Description: Computes the combined Categorical Cross-Entropy loss for start and end index predictions.
    Parameters:
    ground_truth (tf.Tensor): Ground truth tensor of shape [B, max_sequence_length, 2].
    prediction (tf.Tensor): Predicted tensor of shape [B, max_sequence_length, 2].
    Returns:
    combined_loss (tf.Tensor): Combined loss for start and end index predictions, computed as the sum of individual Categorical Cross-Entropy losses weighted by alpha.
    """
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    alpha = 1.0
    start_pred, end_pred = get_start_end_arrays(prediction)
    start_gt, end_gt = get_start_end_arrays(ground_truth)
    combined_loss = loss(start_gt, start_pred) + alpha * loss(end_gt, end_pred)
    return combined_loss