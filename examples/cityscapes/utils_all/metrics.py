from typing import Tuple, List

import tensorflow as tf

from project_config import MODEL_FORMAT, IMAGE_SIZE
from utils_all.preprocessing import Cityscapes
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

def convert_to_xyxy(bounding_boxes: List) -> List[List[int]]:
    xyxy_boxes = []
    for box in bounding_boxes:
        center_x, center_y, width, height, label = box.x, box.y, box.width, box.height, box.label
        class_id = Cityscapes.get_class_id(label)
        x_min = (center_x - width / 2) * IMAGE_SIZE[0]
        y_min = (center_y - height / 2) * IMAGE_SIZE[1]
        x_max = (center_x + width / 2) * IMAGE_SIZE[0]
        y_max = (center_y + height / 2) * IMAGE_SIZE[1]
        xyxy_boxes.append([x_min, y_min, x_max, y_max, class_id])
    return xyxy_boxes


def intersection_area(true_box: List[float], pred_box: List[float]) ->float:
  """Calculates the intersection area between two bounding boxes.

  Args:
    true_box: A bounding box in the format [x1, y1, x2, y2].
    pred_box: A bounding box in the format [x1, y1, x2, y2].

  Returns:
    The intersection area.
  """

  x1 = max(true_box[0], pred_box[0])
  y1 = max(true_box[1], pred_box[1])
  x2 = min(true_box[2], pred_box[2])
  y2 = min(true_box[3], pred_box[3])
  if x2 < x1 or y2 < y1:
    return 0
  else:
    return (x2 - x1) * (y2 - y1)

def union_area(true_box: List[float], pred_box: List[float]) ->float:
  """Calculates the union area between two bounding boxes.

  Args:
    true_box: A bounding box in the format [x1, y1, x2, y2].
    pred_box: A bounding box in the format [x1, y1, x2, y2].

  Returns:
    The union area.
  """

  true_area = (true_box[2] - true_box[0]) * (true_box[3] - true_box[1])
  pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
  return true_area + pred_area - intersection_area(true_box, pred_box)

def calculate_iou(y_true: List[List[float]], y_pred: List[List[float]]) -> tf.Tensor:
  """Calculates the intersection over union (IoU) between a list of true bounding boxes and a list of predicted bounding boxes.

  Args:
    true_boxes: A list of y bounding boxes in the format [x1, y1, x2, y2].
    pred_boxes: A list of y bounding boxes in the format [x1, y1, x2, y2].

  Returns:
    A list of y IoU scores.
  """

  iou_scores = []
  for true_box, pred_box in zip(y_true, y_pred):
    intersection = intersection_area(true_box, pred_box)
    union = union_area(true_box, pred_box)
    # Calculate the IOU value
    iou = tf.where(union > 0, intersection / union, 0)
    iou_scores.append(iou)

  return iou_scores

