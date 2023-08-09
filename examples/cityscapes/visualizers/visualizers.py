from typing import List
import numpy as np
import tensorflow as tf

from config import CONFIG
from utils_all.general_utils import bb_array_to_object, get_predict_bbox_list
from utils_all.preprocessing import CATEGORIES_no_background

from code_loader.contract.responsedataclasses import BoundingBox
from code_loader.contract.visualizer_classes import LeapImageWithBBox


def gt_bb_decoder(image: np.ndarray, bb_gt: tf.Tensor) -> LeapImageWithBBox:
    """
    This function overlays ground truth bounding boxes (BBs) on the input image.

    Parameters:
    image (np.ndarray): The input image for which the ground truth bounding boxes need to be overlaid.
    bb_gt (np.ndarray): The ground truth bounding box array for the input image.

    Returns:
    An instance of LeapImageWithBBox containing the input image with ground truth bounding boxes overlaid.
    """
    bb_object: List[BoundingBox] = bb_array_to_object(bb_gt, iscornercoded=False, bg_label=CONFIG['BACKGROUND_LABEL'],
                                                      is_gt=True)
    bb_object = [bbox for bbox in bb_object if bbox.label in CATEGORIES_no_background]
    return LeapImageWithBBox(data=(image * 255).astype(np.float32), bounding_boxes=bb_object)


def bb_car_gt_decoder(image: np.ndarray, bb_gt: tf.Tensor) -> LeapImageWithBBox:
    """
    Overlays the BB predictions on the image
    """
    bb_object: List[BoundingBox] = bb_array_to_object(bb_gt, iscornercoded=False, bg_label=CONFIG['BACKGROUND_LABEL'], is_gt=True)
    bb_object = [bbox for bbox in bb_object if bbox.label == 'car']
    return LeapImageWithBBox(data=(image * 255).astype(np.float32), bounding_boxes=bb_object)

def bb_decoder(image: np.ndarray, predictions: tf.Tensor) -> LeapImageWithBBox:
    """
    Overlays the BB predictions on the image
    """
    bb_object = get_predict_bbox_list(predictions)
    bb_object = [bbox for bbox in bb_object if bbox.label in CATEGORIES_no_background]
    return LeapImageWithBBox(data=(image * 255).astype(np.float32), bounding_boxes=bb_object)

def bb_car_decoder(image: np.ndarray, predictions: tf.Tensor) -> LeapImageWithBBox:
    """
    Overlays the BB predictions on the image
    """
    bb_object = get_predict_bbox_list(predictions)
    bb_object = [bbox for bbox in bb_object if bbox.label == 'car']
    return LeapImageWithBBox(data=(image * 255).astype(np.float32), bounding_boxes=bb_object)



