
from copy import deepcopy
from typing import Union, Optional, List, Dict

import numpy as np
import tensorflow as tf
from code_loader.contract.responsedataclasses import BoundingBox
from code_loader.helpers.detection.utils import xyxy_to_xywh_format, xywh_to_xyxy_format
from code_loader.helpers.detection.yolo.utils import reshape_output_list
from numpy._typing import NDArray

from cityscapes.preprocessing import BATCH_SIZE, CATEGORIES, MODEL_FORMAT, BACKGROUND_LABEL, image_size, \
    MAX_INSTANCES_PER_CLASS, MAX_BB_PER_IMAGE
from cityscapes.yolo_helpers.yolo_utils import DECODER, DEFAULT_BOXES


#todo
#CATEGORIES??

def polygon_to_bbox(polygon): #TODO: change description
    """
    Converts a polygon representation to a bounding box representation.

    Args:
        vertices (list): List of vertices defining the polygon. The vertices should be in the form [x1, y1, x2, y2, ...].

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


def calculate_iou_all_pairs(bboxes: np.ndarray, image_size: int) -> np.ndarray:
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
        Union[float, int]:
    img_size = img.shape[0]
    label = CATEGORIES.index('Object')
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
        bb_array = bb_array.numpy()
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
                                  label=CATEGORIES[int(bb_array[i][min(5, len(bb_array[i]) - 1)])])

            bb_list.append(curr_bb)
    return bb_list

#TODO: what is this name?

def get_mask_list(data, is_gt):
    is_inference = MODEL_FORMAT == "inference"
    if is_gt:
        bb_object = bb_array_to_object(data, iscornercoded=False, bg_label=BACKGROUND_LABEL, is_gt=True)
    else:
        from_logits = not is_inference
        decoded = is_inference
        class_list_reshaped, loc_list_reshaped = reshape_output_list(
            np.reshape(data, (1, *data.shape)), decoded=decoded, image_size=image_size)
        # add batch
        outputs = DECODER(loc_list_reshaped,
                          class_list_reshaped,
                          DEFAULT_BOXES,
                          from_logits=from_logits,
                          decoded=decoded,
                          )
        bb_object = bb_array_to_object(outputs[0], iscornercoded=True, bg_label=BACKGROUND_LABEL)
    return bb_object


def remove_label_from_bbs(bbs_object_array, removal_label, add_to_label):
    new_bb_arr = []
    for bb in bbs_object_array:
        if bb.label != removal_label:
            new_bb = deepcopy(bb)
            new_bb.label = new_bb.label + "_" + add_to_label
            new_bb_arr.append(new_bb)
    return new_bb_arr


def calculate_overlap(box1, box2):
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


def get_argmax_map(image, bbs):
    image_size = image.shape[:2]
    argmax_map = np.zeros(image_size, dtype=np.uint8)
    cats_dict = {}
    for bb in zip(bbs):
        label = bb.label
        instance_number = cats_dict.get(label, 0)
        # update counter if reach max instances we treat the last objects as one
        cats_dict[label] = instance_number + 1 if instance_number < MAX_INSTANCES_PER_CLASS else instance_number
        #TODO: #argmax_map[resize_mask] = CATEGORIES.index(label) * MAX_INSTANCES_PER_CLASS + cats_dict[label]  # curr_idx
    argmax_map[argmax_map == 0] = len(INSTANCES) + 1
    argmax_map -= 1
    return {"argmax_map": argmax_map}

#TODO:
def extract_bboxes(idx: int, data: Dict):
    x = data['samples'][idx]
    coco = data['cocofile']
    ann_ids = coco.getAnnIds(imgIds=x['id'])
    anns = coco.loadAnns(ann_ids)
    bboxes = np.zeros([MAX_BB_PER_IMAGE, 5])
    max_anns = min(MAX_BB_PER_IMAGE, len(anns))
    for i in range(max_anns):
        ann = anns[i]
        img_size = (x['height'], x['width'])
        class_id = 2 - ann['category_id']
        bbox = polygon_to_bbox(ann['segmentation'][0])
        bbox /= np.array((img_size[1], img_size[0], img_size[1], img_size[0]))
        bboxes[i, :4] = bbox
        bboxes[i, 4] = class_id
    bboxes[max_anns:, 4] = BACKGROUND_LABEL
    return bboxes