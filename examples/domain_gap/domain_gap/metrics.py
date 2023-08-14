import tensorflow as tf

from domain_gap.data.cs_data import CATEGORIES


def class_mean_iou(y_true, y_pred) -> dict:
    """
    Calculate the mean Intersection over Union (mIOU) for segmentation using TensorFlow.

    Args:
        y_true (tf.Tensor): Ground truth segmentation mask tensor.
        y_pred (tf.Tensor): Predicted segmentation mask tensor.

    Returns:
        tf.Tensor: Mean Intersection over Union (mIOU) value.
    """
    res = {}
    for i, c in enumerate(CATEGORIES):
        y_true_i, y_pred_i = y_true[..., i], y_pred[..., i]
        res[f'{c}'] = mean_iou(y_true_i, y_pred_i)
    return res


def get_class_mean_iou(class_i: int = None):

    def class_mean_iou(y_true, y_pred):
        """
        Calculate the mean Intersection over Union (mIOU) for segmentation using TensorFlow.

        Args:
            y_true (tf.Tensor): Ground truth segmentation mask tensor.
            y_pred (tf.Tensor): Predicted segmentation mask tensor.

        Returns:
            tf.Tensor: Mean Intersection over Union (mIOU) value.
        """
        y_true, y_pred = y_true[..., class_i], y_pred[..., class_i]
        iou = mean_iou(y_true, y_pred)

        return iou

    return class_mean_iou


def mean_iou(y_true, y_pred):
    """
    Calculate the mean Intersection over Union (mIOU) for segmentation using TensorFlow.

    Args:
        y_true (tf.Tensor): Ground truth segmentation mask tensor.
        y_pred (tf.Tensor): Predicted segmentation mask tensor.

    Returns:
        tf.Tensor: Mean Intersection over Union (mIOU) value.
    """
    # Flatten the tensors
    y_true_flat = tf.reshape(y_true, [y_true.shape[0], -1])
    y_pred_flat = tf.cast(tf.reshape(y_pred, [y_true.shape[0], -1]), y_true_flat.dtype)

    # Calculate the intersection and union
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat, -1)
    union = tf.reduce_sum(tf.maximum(y_true_flat, y_pred_flat), -1)

    # Calculate the IOU value
    iou = tf.where(union > 0, intersection / union, 0)

    return iou