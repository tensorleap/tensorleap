from typing import Tuple
import tensorflow as tf

from project_config import MODEL_FORMAT, IMAGE_SIZE
from yolo_helpers.yolo_utils import LOSS_FN

from code_loader.helpers.detection.yolo.utils import reshape_output_list

def compute_losses(obj_true: tf.Tensor, od_pred: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Computes the sum of the classification (CE loss) and localization (regression) losses from all heads
    """
    decoded = False if MODEL_FORMAT != "inference" else True
    class_list_reshaped, loc_list_reshaped = reshape_output_list(od_pred, decoded=decoded,
                                                                 image_size=IMAGE_SIZE)  # add batch
    loss_l, loss_c, loss_o = LOSS_FN(y_true=obj_true, y_pred=(loc_list_reshaped, class_list_reshaped))

    return loss_l, loss_c, loss_o

def od_loss(bb_gt: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:  # return batch
    """
    Sums the classification and regression loss
    """
    loss_l, loss_c, loss_o = compute_losses(bb_gt, y_pred)
    combined_losses = [l + c + o for l, c, o in zip(loss_l, loss_c, loss_o)]
    sum_loss = tf.reduce_sum(combined_losses, axis=0)
    non_nan_loss = tf.where(tf.math.is_nan(sum_loss), tf.zeros_like(sum_loss), sum_loss) #LOSS 0 for NAN losses
    return non_nan_loss

def classification_metric(bb_gt: tf.Tensor, detection_pred: tf.Tensor) -> tf.Tensor:  # return batch
    """
    This function calculates the total classification loss for each head of the object detection model.

    Parameters:
    bb_gt (tf.Tensor): The ground truth tensor containing the target bounding box values.
    detection_pred (tf.Tensor): The predicted tensor containing the output from the object detection model.

    Returns:
    A tensor representing the total classification (cross-entropy) loss for each head.
    """
    _, loss_c, _ = compute_losses(bb_gt, detection_pred)
    return tf.reduce_sum(loss_c, axis=0)[:, 0]


def regression_metric(bb_gt: tf.Tensor, detection_pred: tf.Tensor) -> tf.Tensor:  # return batch
    """
    This function calculates the total regression (localization) loss for each head of the object detection model.
    Parameters:
    bb_gt (tf.Tensor): The ground truth tensor containing the target bounding box values.
    detection_pred (tf.Tensor): The predicted tensor containing the output from the object detection model.

    Returns:
    A tensor representing the total regression (localization) loss for each head.
    """
    loss_l, _, _ = compute_losses(bb_gt, detection_pred)
    return tf.reduce_sum(loss_l, axis=0)[:, 0]  # shape of batch


def object_metric(bb_gt: tf.Tensor, detection_pred: tf.Tensor) -> tf.Tensor:
    """
    This function calculates the total objectness loss for each head of the object detection model.

    Parameters:
    bb_gt (tf.Tensor): The ground truth tensor containing the target bounding box values.
    detection_pred (tf.Tensor): The predicted tensor containing the output from the object detection model.

    Returns:
    A tensor representing the total objectness loss for each head.
    """
    _, _, loss_o = compute_losses(bb_gt, detection_pred)
    return tf.reduce_sum(loss_o, axis=0)[:, 0]  # shape of batch
