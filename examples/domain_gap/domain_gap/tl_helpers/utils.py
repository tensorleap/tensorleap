import json
from typing import Dict
import numpy as np
from code_loader.contract.datasetclasses import PreprocessResponse
from PIL import Image
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops

from domain_gap.utils.gcs_utils import _download
from domain_gap.data.cs_data import Cityscapes
from domain_gap.utils.configs import IMAGE_SIZE, AUGMENT, TRAIN_SIZE, NUM_CLASSES


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

def get_categorical_mask(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data
    cloud_path = data['gt_path'][idx % data["real_size"]]
    fpath = _download(cloud_path)
    mask = np.array(Image.open(fpath).resize(IMAGE_SIZE, Image.Resampling.NEAREST))
    if data['dataset'][idx % data["real_size"]] == 'cityscapes':
        encoded_mask = Cityscapes.encode_target_cityscapes(mask)
    else:
        encoded_mask = Cityscapes.encode_target(mask)
    return encoded_mask


def get_metadata_json(idx: int, data: PreprocessResponse) -> Dict[str, str]:
    cloud_path = data.data['metadata'][idx]
    fpath = _download(cloud_path)
    with open(fpath, 'r') as f:
        metadata_dict = json.loads(f.read())
    return metadata_dict


def aug_factor_or_zero(idx: int, data: PreprocessResponse, value: float) -> float:
    if data.data["subset_name"] == "train" and AUGMENT and idx > TRAIN_SIZE - 1:
        return value.numpy()
    else:
        return 0.