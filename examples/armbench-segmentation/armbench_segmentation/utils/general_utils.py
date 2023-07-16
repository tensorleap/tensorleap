from copy import deepcopy
from typing import Union, Optional, List, Dict

import numpy as np
import tensorflow as tf
from code_loader.contract.responsedataclasses import BoundingBox
from code_loader.helpers.detection.utils import xyxy_to_xywh_format, xywh_to_xyxy_format
from code_loader.helpers.detection.yolo.utils import reshape_output_list
from numpy._typing import NDArray

from armbench_segmentation import CACHE_DICTS
from armbench_segmentation.config import CONFIG
from armbench_segmentation.yolo_helpers.yolo_utils import DECODER, DEFAULT_BOXES


def polygon_to_bbox(vertices):
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

    xs = [x for i, x in enumerate(vertices) if i % 2 == 0]
    ys = [x for i, x in enumerate(vertices) if i % 2 != 0]

    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xs)
    max_y = max(ys)

    # Bounding box representation: (x, y, width, height)
    bbox = [(min_x + max_x) / 2., (min_y + max_y) / 2., max_x - min_x, max_y - min_y]

    return bbox


def calculate_iou_all_pairs(bboxes: np.ndarray, image_size: int) -> np.ndarray:
    """
    Calculates the Intersection over Union (IOU) for all pairs of bounding boxes.

    This function utilizes vectorization to efficiently compute the IOU for all possible pairs of bounding boxes.
    By leveraging NumPy's array operations, the calculations are performed in parallel, leading to improved performance.

    Args:
        bboxes (np.ndarray): Array of bounding boxes in the format [x, y, w, h].
        image_size (int): Size of the image.

    Returns:
        np.ndarray: Array containing the IOU values for all pairs of bounding boxes.
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
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

    # Calculate intersection area for all pairs
    intersection_area = np.maximum(x_max - x_min, 0) * np.maximum(y_max - y_min, 0)

    # Calculate union area for all pairs
    union_area = areas[:, np.newaxis] + areas - intersection_area

    # Calculate IOU for all pairs
    iou = intersection_area / union_area
    iou = iou[np.triu_indices(num_bboxes, k=1)]
    return iou


def count_obj_bbox_occlusions(img: np.ndarray, bboxes: np.ndarray, occlusion_threshold: float, calc_avg_flag: bool) -> \
        Union[float, int]:
    """
    Counts the occluded bounding boxes of a specific object category in an image.

    This function takes an image and an array of bounding boxes as input and counts the number of occluded
    bounding boxes of a specific object category. The occlusion is determined based on the Intersection over Union (IOU)
    between the bounding boxes.

    Args:
        img (np.ndarray): Image represented as a NumPy array.
        bboxes (np.ndarray): Array of bounding boxes in the format [x, y, w, h, label].
        occlusion_threshold (float): Threshold value for determining occlusion based on IOU.
        calc_avg_flag (bool): Flag indicating whether to calculate the average occlusion count.

    Returns:
        Union[float, int]: Number of occluded bounding boxes of the specified object category.
                           If calc_avg_flag is True, it returns the average occlusion count as a float.
                           If calc_avg_flag is False, it returns the total occlusion count as an integer.

    """
    img_size = img.shape[0]
    label = CONFIG["CATEGORIES"].index('Object')
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


def count_obj_masks_occlusions(masks: Union[np.ndarray, None], occlusion_threshold: float) -> int:
    """
    Counts the occluded masks based on Intersection over Union (IOU).

    Args:
        masks (Union[np.ndarray, None]): Masks represented as a NumPy array. Can be None if no masks are provided.
        occlusion_threshold (float): Threshold value for determining occlusion based on IOU.

    Returns:
        int: Number of occluded masks based on the specified occlusion threshold.

    """

    if masks is None:
        return 0

    if masks[0, ...].shape != CONFIG["IMAGE_SIZE"]:
        masks = tf.image.resize(masks[..., None], CONFIG["IMAGE_SIZE"], tf.image.ResizeMethod.NEAREST_NEIGHBOR)[..., 0]
        masks = masks.numpy()

    num_masks = len(masks)

    # Reshape masks to have a third dimension
    masks = np.expand_dims(masks, axis=-1)

    # Create tiled versions of masks for element-wise addition
    tiled_masks = np.broadcast_to(masks, (num_masks, num_masks, masks.shape[1], masks.shape[2], 1))
    tiled_masks_transposed = np.transpose(tiled_masks, axes=(1, 0, 2, 3, 4))

    # Compute overlay matrix
    overlay = tiled_masks + tiled_masks_transposed

    # Exclude same mask occlusions and duplicate pairs
    mask_indices = np.triu_indices(num_masks, k=1)
    overlay = overlay[mask_indices]

    intersection = np.sum(overlay > 1, axis=(-1, -2, -3))
    union = np.sum(overlay > 0, axis=(-1, -2, -3))

    iou = intersection / union
    return int(np.sum(iou > occlusion_threshold))


def bb_array_to_object(bb_array: Union[NDArray[float], tf.Tensor], iscornercoded: bool = True, bg_label: int = 0,
                       is_gt=False, masks: Optional[tf.Tensor] = None) -> List[BoundingBox]:
    """
    Converts a bounding box array to a list of BoundingBox objects.

    This function takes a bounding box array and optional masks as input and converts it into a list of BoundingBox objects.
    The bounding box array is expected to be in the format (CLASSES, TOP_K, PROPERTIES), where PROPERTIES corresponds to
    (confidence, xmin, ymin, xmax, ymax). If iscornercoded is set to True, the bounding box coordinates are assumed to be
    in the format (xmin, ymin, xmax, ymax); otherwise, they are assumed to be in the format (x, y, width, height).

    Args:
        bb_array (Union[NDArray[float], tf.Tensor]): Bounding box array.
        iscornercoded (bool, optional): Flag indicating whether the bounding box coordinates are corner-coded.
                                        Defaults to True.
        bg_label (int, optional): Background label index. Defaults to 0.
        is_gt (bool, optional): Flag indicating whether the bounding box array represents ground truth.
                                Defaults to False.
        masks (Optional[tf.Tensor], optional): Optional masks associated with the bounding boxes. Defaults to None.

    Returns:
        Tuple[List[BoundingBox], List[np.ndarray]]: Tuple containing the list of BoundingBox objects and the list of masks.
                                                    If masks are not provided, the second element will be an empty list.

    """

    bb_list = []
    mask_list = []
    if not isinstance(bb_array, np.ndarray):
        bb_array = bb_array.numpy()
    if len(bb_array.shape) == 3:
        bb_array = bb_array.reshape(-1, bb_array.shape[-1])
    if masks is not None and len(bb_array) > 0:
        if not isinstance(masks, np.ndarray):
            np_masks = masks.numpy()
        else:
            np_masks = masks
        if not is_gt:
            out_masks = (tf.sigmoid(np_masks @ bb_array[:, 6:].T).numpy() > 0.5).astype(float)
        else:
            out_masks = np.swapaxes(np.swapaxes(np_masks, 0, 1), 1, 2)
        no_object_masks = np.zeros_like(out_masks)
        h_factor = out_masks.shape[0]
        w_factor = out_masks.shape[1]
    for i in range(bb_array.shape[0]):
        if bb_array[i][-1] != bg_label:
            if iscornercoded:
                x, y, w, h = xyxy_to_xywh_format(bb_array[i][1:5])  # FIXED TOM
            else:
                x, y = bb_array[i][0], bb_array[i][1]
                w, h = bb_array[i][2], bb_array[i][3]
            conf = 1 if is_gt else bb_array[i][0]
            curr_bb = BoundingBox(x=x, y=y, width=w, height=h, confidence=conf,
                                  label=CONFIG["CATEGORIES"][int(bb_array[i][min(5, len(bb_array[i]) - 1)])])
            if masks is not None:
                if is_gt:
                    fixed_mask = out_masks[..., i]
                else:
                    no_object_masks[
                    int(max(y - h / 2, 0) * h_factor):int(min(y + h / 2, out_masks.shape[1]) * h_factor),
                    int(max(x - w / 2, 0) * h_factor):int(min(x + w / 2, out_masks.shape[1]) * w_factor), i] = 1
                    fixed_mask = out_masks[..., i] * no_object_masks[..., i]
                mask_list.append(fixed_mask.round().astype(int))
            bb_list.append(curr_bb)
    return bb_list, mask_list


def get_mask_list(data, masks, is_gt):
    res = CACHE_DICTS['mask_list'].get(str(data) + str(masks) + str(is_gt))
    if res is not None:
        return res

    is_inference = CONFIG["MODEL_FORMAT"] == "inference"
    if is_gt:
        bb_object, mask_list = bb_array_to_object(data, iscornercoded=False, bg_label=CONFIG["BACKGROUND_LABEL"],
                                                  is_gt=True,
                                                  masks=masks)
    else:
        from_logits = not is_inference
        decoded = is_inference
        class_list_reshaped, loc_list_reshaped = reshape_output_list(
            np.reshape(data, (1, *data.shape)), decoded=decoded, image_size=CONFIG["IMAGE_SIZE"])
        outputs = DECODER(loc_list_reshaped,
                          class_list_reshaped,
                          DEFAULT_BOXES,
                          from_logits=from_logits,
                          decoded=decoded,
                          )
        bb_object, mask_list = bb_array_to_object(outputs[0], iscornercoded=True, bg_label=CONFIG["BACKGROUND_LABEL"],
                                                  masks=masks)
    if len(CACHE_DICTS['mask_list'].keys()) > 4 * CONFIG["BATCH_SIZE"]:
        CACHE_DICTS['mask_list'] = {str(data) + str(masks) + str(is_gt): (bb_object, mask_list)}
    else:
        CACHE_DICTS['mask_list'][str(data) + str(masks) + str(is_gt)] = (bb_object, mask_list)
    return bb_object, mask_list


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


def get_argmax_map_and_separate_masks(image, bbs, masks):
    image_size = image.shape[:2]
    argmax_map = np.zeros(image_size, dtype=np.uint8)
    cats_dict = {}
    separate_masks = []
    for bb, mask in zip(bbs, masks):
        if mask.shape != image_size:
            resize_mask = tf.image.resize(mask[..., None], image_size, tf.image.ResizeMethod.NEAREST_NEIGHBOR)[..., 0]
            if not isinstance(resize_mask, np.ndarray):
                resize_mask = resize_mask.numpy()
        else:
            resize_mask = mask
        resize_mask = resize_mask.astype(bool)
        label = bb.label
        instance_number = cats_dict.get(label, 0)
        # update counter if reach max instances we treat the last objects as one
        cats_dict[label] = instance_number + 1 if instance_number < CONFIG["MAX_INSTANCES_PER_CLASS"] else instance_number
        argmax_map[resize_mask] = CONFIG["CATEGORIES"].index(label) * CONFIG["MAX_INSTANCES_PER_CLASS"] + cats_dict[label]  # curr_idx
        if bb.label == 'Object':
            separate_masks.append(resize_mask)
    argmax_map[argmax_map == 0] = len(CONFIG['INSTANCES']) + 1
    argmax_map -= 1
    return {"argmax_map": argmax_map, "separate_masks": separate_masks}


def extract_and_cache_bboxes(idx: int, data: Dict):
    res = CACHE_DICTS['bbs'].get(str(idx) + data['subdir'])
    if res is not None:
        return res
    x = data['samples'][idx]
    coco = data['cocofile']
    ann_ids = coco.getAnnIds(imgIds=x['id'])
    anns = coco.loadAnns(ann_ids)
    bboxes = np.zeros([CONFIG['MAX_BB_PER_IMAGE'], 5])
    max_anns = min(CONFIG['MAX_BB_PER_IMAGE'], len(anns))
    for i in range(max_anns):
        ann = anns[i]
        img_size = (x['height'], x['width'])
        class_id = 2 - ann['category_id']
        bbox = polygon_to_bbox(ann['segmentation'][0])
        bbox /= np.array((img_size[1], img_size[0], img_size[1], img_size[0]))
        bboxes[i, :4] = bbox
        bboxes[i, 4] = class_id
    bboxes[max_anns:, 4] = CONFIG['BACKGROUND_LABEL']
    if len(CACHE_DICTS['bbs'].keys()) > CONFIG['BATCH_SIZE']:
        CACHE_DICTS['bbs'] = {str(idx) + data['subdir']: bboxes}
    else:
        CACHE_DICTS['bbs'][str(idx) + data['subdir']] = bboxes
    return bboxes
