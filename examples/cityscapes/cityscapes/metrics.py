from typing import Tuple, List, Union
import tensorflow as tf

from code_loader.helpers.detection.yolo.utils import reshape_output_list

from cityscapes.preprocessing import MODEL_FORMAT, image_size
from cityscapes.yolo_helpers.yolo_utils import LOSS_FN

#TODO: over it

def compute_losses(obj_true: tf.Tensor, od_pred: tf.Tensor,
                   mask_true, instance_seg: tf.Tensor) -> Union[
    Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]],
    Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]]]:
    """
    Computes the sum of the classification (CE loss) and localization (regression) losses from all heads
    """
    decoded = False if MODEL_FORMAT != "inference" else True
    class_list_reshaped, loc_list_reshaped = reshape_output_list(od_pred, decoded=decoded,
                                                                 image_size=image_size)  # add batch
    loss_l, loss_c, loss_o, loss_m = LOSS_FN(y_true=obj_true, y_pred=(loc_list_reshaped, class_list_reshaped),
                                             instance_seg=instance_seg, instance_true=mask_true)

    return loss_l, loss_c, loss_o, loss_m

def classification_metric(bb_gt: tf.Tensor, detection_pred: tf.Tensor,
                          mask_gt: tf.Tensor, segmentation_pred: tf.Tensor):  # return batch
    _, loss_c, _, _ = compute_losses(bb_gt, detection_pred, mask_gt, segmentation_pred)
    return tf.reduce_sum(loss_c, axis=0)[:, 0]


def regression_metric(bb_gt: tf.Tensor, detection_pred: tf.Tensor,
                      mask_gt: tf.Tensor, segmentation_pred: tf.Tensor):  # return batch
    loss_l, _, _, _ = compute_losses(bb_gt, detection_pred, mask_gt, segmentation_pred)
    return tf.reduce_sum(loss_l, axis=0)[:, 0]  # shape of batch


def object_metric(bb_gt: tf.Tensor, detection_pred: tf.Tensor,
                  mask_gt: tf.Tensor, segmentation_pred: tf.Tensor):
    _, _, loss_o, _ = compute_losses(bb_gt, detection_pred, mask_gt, segmentation_pred)
    return tf.reduce_sum(loss_o, axis=0)[:, 0]  # shape of batch

