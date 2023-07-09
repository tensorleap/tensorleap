
import tensorflow as tf

def get_start_end_arrays(array):
    start_arr = array[..., 0]
    end_arr = array[..., 1]
    return start_arr, end_arr

def exact_match_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:  # return batch
    """
    This checks if the prediction is identical for the gt and prediction
    y_true :[B, max_sequence_length, 2]
    y_pred: [B, max_sequence_length, 2]
    """
    is_argmax_equal = tf.equal(tf.argmax(y_true, axis=1), tf.argmax(y_pred, axis=1))
    is_exact_match = tf.reduce_all(is_argmax_equal, axis=1)
    return tf.cast(is_exact_match, tf.float64)


def get_nonnegative_tensor(tensor: tf.Tensor) -> tf.Tensor:
    return tf.where(tensor > 0, tensor, tf.zeros_like(tensor))


def f1_metric(y_true: tf.Tensor, y_pred: tf.Tensor):  # return batch
    """
    Computes the F1 metric (2*precision*recall)/(precision + recall)
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
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    start_gt, end_gt = get_start_end_arrays(ground_truth)
    start_pred, end_pred = get_start_end_arrays(prediction)
    return loss(start_gt, start_pred)

def CE_end_index(ground_truth: tf.Tensor, prediction: tf.Tensor) -> tf.Tensor:
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    start_gt, end_gt = get_start_end_arrays(ground_truth)
    start_pred, end_pred = get_start_end_arrays(prediction)
    return loss(end_gt, end_pred)