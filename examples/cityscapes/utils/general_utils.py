from copy import deepcopy
from typing import Union, List
from numpy._typing import NDArray
import numpy as np
import tensorflow as tf

from cityscapes.preprocessing import CATEGORIES, Cityscapes

from code_loader.contract.responsedataclasses import BoundingBox
from code_loader.helpers.detection.utils import xyxy_to_xywh_format, xywh_to_xyxy_format

def extract_bounding_boxes_from_instance_segmentation_polygons(json_data):
    """
    This function extracts bounding boxes from instance segmentation polygons present in the given JSON data.
    :param json_data: (dict) A dictionary containing instance segmentation polygons and image size information.
    :return: bounding_boxes: (numpy.ndarray) An array of bounding boxes in the format [x, y, width, height, class_id].
    """
    objects = json_data['objects']
    bounding_boxes = []
    image_size = (json_data['imgHeight'], json_data['imgWidth'])
    for object in objects:
        b = np.zeros(5)
        class_label = object['label']
        class_id = Cityscapes.get_class_id(class_label)
        if class_id is None:
            continue
        else:
            bbox = polygon_to_bbox(object['polygon'])
            bbox /= np.array((image_size[1], image_size[0], image_size[1], image_size[0]))
            b[:4] = bbox
            b[4] = class_id
            bounding_boxes.append(b)
    bounding_boxes = np.array(bounding_boxes)
    return bounding_boxes

def polygon_to_bbox(polygon): #TODO: change description
    """
    Converts a polygon representation to a bounding box representation.

    Args:
        vertices: (list) List of vertices defining the polygon. The vertices should be in the form [x1, y1, x2, y2, ...].

    Returns:
        list: Bounding box representation of the polygon in the form [x, y, width, height].

    Note:
        - The input list of vertices should contain x and y coordinates in alternating order.
        - The function calculates the minimum and maximum values of the x and y coordinates to determine the bounding box.
        - The bounding box representation is returned as [x, y, width, height], where (x, y) represents the center point of the
          bounding box, and width and height denote the size of the bounding box.
    """

    min_x = min(x for x, y in polygon)
    min_y = min(y for x, y in polygon)
    max_x = max(x for x, y in polygon)
    max_y = max(y for x, y in polygon)

    # Bounding box representation: (x, y, width, height)
    bbox = [(min_x + max_x) / 2., (min_y + max_y) / 2., max_x - min_x, max_y - min_y]
    return bbox


def calculate_iou_all_pairs(bboxes: np.ndarray, image_size: int) -> np.ndarray: #TODO: no use
    """
    This function calculates the Intersection over Union (IOU) for all pairs of bounding boxes.
    :param bboxes: (numpy.ndarray) An array of bounding boxes in the format [x, y, width, height, class_id].
    :param image_size: (int) Size of the image (assumed to be square).
    :return: iou (numpy.ndarray) An array containing the calculated IOU values for all pairs of bounding boxes.
    """
    # Reformat all bboxes to (x_min, y_min, x_max, y_max)
    bboxes = np.asarray([xywh_to_xyxy_format(bbox[:-1]) for bbox in bboxes]) * image_size
    num_bboxes = len(bboxes)
    # Calculate coordinates for all pairs
    x_min = np.maximum(bboxes[:, 0][:, np.newaxis], bboxes[:, 0])
    y_min = np.maximum(bboxes[:, 1][:, np.newaxis], bboxes[:, 1])
    x_max = np.minimum(bboxes[:, 2][:, np.newaxis], bboxes[:, 2])
    y_max = np.minimum(bboxes[:, 3][:, np.newaxis], bboxes[:, 3])

    # Calculate areas for all pairs
    area_a = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    area_b = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

    # Calculate intersection area for all pairs
    intersection_area = np.maximum(x_max - x_min, 0) * np.maximum(y_max - y_min, 0)

    # Calculate union area for all pairs
    union_area = area_a[:, np.newaxis] + area_b - intersection_area

    # Calculate IOU for all pairs
    iou = intersection_area / union_area
    iou = iou[np.triu_indices(num_bboxes, k=1)]
    return iou


def count_obj_bbox_occlusions(img: np.ndarray, bboxes: np.ndarray, occlusion_threshold: float, calc_avg_flag: bool) -> \
        Union[float, int]: #TODO: no use
    """
    This function counts the number of occlusions for a specific object class within the given image.
    :param img: (numpy.ndarray) The image array.
    :param bboxes: (numpy.ndarray) An array of bounding boxes in the format [x, y, width, height, class_id].
    :param occlusion_threshold: (float) IOU threshold for considering a bounding box as occluded.
    :param calc_avg_flag: (bool) If True, returns the average occlusion count per object of the specified class.
    :return: If calc_avg_flag is True, the function returns the average occlusion count per object as a float.
             If calc_avg_flag is False, the function returns the total occlusion count for the specified object class as an integer.
    """
    img_size = img.shape[0]
    label = CATEGORIES.index('Object') #TODO: change
    obj_bbox = bboxes[bboxes[..., -1] == label]
    if len(obj_bbox) == 0:
        return 0.0
    else:
        ious = calculate_iou_all_pairs(obj_bbox, img_size)
        occlusion_count = len(ious[ious > occlusion_threshold])
        if calc_avg_flag:
            return int(occlusion_count / len(obj_bbox))
        else:
            return occlusion_count


def bb_array_to_object(bb_array: Union[NDArray[float], tf.Tensor], iscornercoded: bool = True, bg_label: int = 0,
                       is_gt=False) -> List[BoundingBox]:
    """
    Assumes a (X,Y,W,H) Format for the BB text
    bb_array is (CLASSES,TOP_K,PROPERTIES) WHERE PROPERTIES =(conf,xmin,ymin,xmax,ymax)
    """
    bb_list = []
    if not isinstance(bb_array, np.ndarray):
        bb_array = np.array(bb_array)
    # fig, ax = plt.subplots(figsize=(6, 9)
    if len(bb_array.shape) == 3:
        bb_array = bb_array.reshape(-1, bb_array.shape[-1])
    for i in range(bb_array.shape[0]):
        if bb_array[i][-1] != bg_label:
            if iscornercoded:
                x, y, w, h = xyxy_to_xywh_format(bb_array[i][1:5])  # FIXED TOM
                # unormalize to image dimensions
            else:
                x, y = bb_array[i][0], bb_array[i][1]
                w, h = bb_array[i][2], bb_array[i][3]
            conf = 1 if is_gt else bb_array[i][0]
            curr_bb = BoundingBox(x=x, y=y, width=w, height=h, confidence=conf,
                                  label=CATEGORIES[int(bb_array[i][min(5, len(bb_array[i]) - 1)])]) #TODO: change maybe

            bb_list.append(curr_bb)
    return bb_list



def remove_label_from_bbs(bbs_object_array, removal_label, add_to_label): #TODO: no use
    """
    This function removes bounding boxes with a specific label and adds a new label suffix to the remaining bounding boxes.
    :param bbs_object_array: An array of BoundingBox objects.
    :param removal_label: The label of the bounding boxes to be removed.
    :param add_to_label: The suffix to be added to the label of the remaining bounding boxes.
    :return: new_bb_arr: A new array of BoundingBox objects after removing and updating labels.

    """
    new_bb_arr = []
    for bb in bbs_object_array:
        if bb.label != removal_label:
            new_bb = deepcopy(bb)
            new_bb.label = new_bb.label + "_" + add_to_label
            new_bb_arr.append(new_bb)
    return new_bb_arr


def calculate_overlap(box1, box2): #TODO: no use
    """
    This function calculates the overlap area between two bounding boxes.
    :param box1: box1 (tuple or list): The first bounding box represented as (x, y, width, height).
    :param box2: box2 (tuple or list): The second bounding box represented as (x, y, width, height).
    :return: overlap_area (float): The area of overlap between the two bounding boxes.
    """
    # Extract coordinates of the bounding boxes
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the coordinates of the intersection rectangle
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = max(0, min(x1 + w1, x2 + w2) - x_intersection)
    h_intersection = max(0, min(y1 + h1, y2 + h2) - y_intersection)

    # Calculate the overlap area
    overlap_area = w_intersection * h_intersection

    return overlap_area