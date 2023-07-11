from copy import deepcopy
from typing import Union, Optional, List

import numpy as np
import tensorflow as tf
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.responsedataclasses import BoundingBox
from code_loader.helpers.detection.utils import xyxy_to_xywh_format, xywh_to_xyxy_format
from code_loader.helpers.detection.yolo.utils import reshape_output_list
from numpy._typing import NDArray

from armbench_segmentation import CACHE_DICTS
from armbench_segmentation.preprocessing import BATCH_SIZE, CATEGORIES, MODEL_FORMAT, BACKGROUND_LABEL, IMAGE_SIZE, \
    DEFAULT_BOXES
from armbench_segmentation.visualizers import multiple_mask_pred, multiple_mask_gt
from armbench_segmentation.yolo_helpers.yolo_utils import DECODER


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


def ioa_mask(mask_containing, mask_contained):
    """
    Calculates the Intersection over Area (IOA) between two binary masks.

    Args:
        mask_containing (ndarray or Tensor): Binary mask representing the containing object.
        mask_contained (ndarray or Tensor): Binary mask representing the contained object.

    Returns:
        float: The IOA (Intersection over Area) value between the two masks.

    Note:
        - The input masks should have compatible shapes.
        - The function performs a bitwise AND operation between the 'mask_containing' and 'mask_contained' masks to obtain
          the intersection mask.
        - It calculates the number of True values in the intersection mask to determine the intersection area.
        - The area of the contained object is computed as the number of True values in the 'mask_contained' mask.
        - If the area of the contained object is 0, the IOA is defined as 0.
        - The IOA value is calculated as the ratio of the intersection area to the maximum of the area of the contained
          object or 1.
    """

    intersection_mask = mask_containing & mask_contained
    intersection = len(intersection_mask[intersection_mask])
    area = len(mask_contained[mask_contained])
    return intersection / max(area, 1)


def get_ioa_array(image, y_pred_bb, y_pred_mask, bb_gt, mask_gt, containing='pred'):
    hash_str = str(image) + str(y_pred_bb) + str(y_pred_mask) + str(bb_gt) + str(mask_gt) + str(containing)
    res = CACHE_DICTS['get_ioa_array'].get(hash_str)
    if res is not None:
        return res
    prediction_masks = multiple_mask_pred(image, y_pred_bb, y_pred_mask)
    gt_masks = multiple_mask_gt(image, bb_gt, mask_gt)
    ioas = np.zeros((len(prediction_masks), len(gt_masks)))
    for i, pred_mask in enumerate(prediction_masks):
        for j, gt_mask in enumerate(gt_masks):
            if containing == 'pred':
                ioas[i, j] = ioa_mask(pred_mask, gt_mask)
            else:
                ioas[i, j] = ioa_mask(gt_mask, pred_mask)
    if len(CACHE_DICTS['get_ioa_array'].keys()) > 2 * BATCH_SIZE:
        CACHE_DICTS['get_ioa_array'] = {hash_str: ioas}
    else:
        CACHE_DICTS['get_ioa_array'][hash_str] = ioas
    return ioas


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


def count_obj_bbox_occlusions(img: np.ndarray, bboxes: np.ndarray, occlusion_threshold: float, calc_avg_flag: bool) -> Union[float, int]:
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


def count_obj_masks_occlusions(masks: Union[np.ndarray, None], occlusion_threshold: float) -> int:
    if masks is None:
        return 0

    if masks[0, ...].shape != IMAGE_SIZE:
        masks = tf.image.resize(masks[..., None], IMAGE_SIZE, tf.image.ResizeMethod.NEAREST_NEIGHBOR)[..., 0]
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
    Assumes a (X,Y,W,H) Format for the BB text
    bb_array is (CLASSES,TOP_K,PROPERTIES) WHERE PROPERTIES =(conf,xmin,ymin,xmax,ymax)
    """
    bb_list = []
    mask_list = []
    if not isinstance(bb_array, np.ndarray):
        bb_array = bb_array.numpy()
    # fig, ax = plt.subplots(figsize=(6, 9)
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
                # unormalize to image dimensions
            else:
                x, y = bb_array[i][0], bb_array[i][1]
                w, h = bb_array[i][2], bb_array[i][3]
            conf = 1 if is_gt else bb_array[i][0]
            curr_bb = BoundingBox(x=x, y=y, width=w, height=h, confidence=conf,
                                  label=CATEGORIES[int(bb_array[i][min(5, len(bb_array[i]) - 1)])])
            if masks is not None:
                if is_gt:
                    fixed_mask = out_masks[..., i]
                else:
                    no_object_masks[
                    int(max(y - h / 2, 0) * h_factor):int(min(y + h / 2, out_masks.shape[1]) * h_factor),
                    int(max(x - w / 2, 0) * h_factor):int(min(x + w / 2, out_masks.shape[1]) * w_factor), i] = 1
                    fixed_mask = out_masks[..., i] * no_object_masks[..., i]
                mask_list.append(fixed_mask.round().astype(int))
                # new_mask = Image.fromarray(fixed_mask).resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.NEAREST)
            # downscale box
            # crop mask according to prediction
            # increase mask to image size
            bb_list.append(curr_bb)
    return bb_list, mask_list


def get_mask_list(data, masks, is_gt):
    res = CACHE_DICTS['mask_list'].get(str(data) + str(masks) + str(is_gt))
    if res is not None:
        return res

    is_inference = MODEL_FORMAT == "inference"
    if is_gt:
        bb_object, mask_list = bb_array_to_object(data, iscornercoded=False, bg_label=BACKGROUND_LABEL,
                                                  is_gt=True,
                                                  masks=masks)
    else:
        from_logits = ~ is_inference
        decoded = is_inference
        class_list_reshaped, loc_list_reshaped = reshape_output_list(
            np.reshape(data, (1, *data.shape)), decoded=decoded, image_size=IMAGE_SIZE)
        # add batch
        outputs = DECODER(loc_list_reshaped,
                          class_list_reshaped,
                          DEFAULT_BOXES,
                          from_logits=from_logits,
                          decoded=decoded,
                          )
        bb_object, mask_list = bb_array_to_object(outputs[0], iscornercoded=True, bg_label=BACKGROUND_LABEL,
                                                  masks=masks)
    if len(CACHE_DICTS['mask_list'].keys()) > 4 * BATCH_SIZE:  # BATCH_SIZE*[FALSE/TRUE,MASK/NO-MASK]
        CACHE_DICTS['mask_list'] = {str(data) + str(masks) + str(is_gt): (bb_object, mask_list)}
    else:
        CACHE_DICTS['mask_list'][str(data) + str(masks) + str(is_gt)] = (bb_object, mask_list)
    return bb_object, mask_list


def get_cat_instances_seg_lst(idx: int, data: PreprocessResponse, cat: str) -> List[np.ma.array]:
    img = input_image(idx, data)
    if cat == "tote":
        masks = get_tote_instances_masks(idx, data)
    elif cat == "object":
        masks = get_object_instances_masks(idx, data)
    else:
        print('Error category not supported')
        return None
    if masks is None:
        return None
    if masks[0, ...].shape != IMAGE_SIZE:
        masks = tf.image.resize(masks[..., None], IMAGE_SIZE, tf.image.ResizeMethod.NEAREST_NEIGHBOR)[..., 0]
        masks = masks.numpy()
    instances = []
    for mask in masks:
        mask = np.broadcast_to(mask[..., np.newaxis], img.shape)
        masked_arr = np.ma.masked_array(img, mask)
        instances.append(masked_arr)
    return instances


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
