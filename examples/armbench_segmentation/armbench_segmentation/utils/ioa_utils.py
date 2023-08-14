import numpy as np

from armbench_segmentation import CACHE_DICTS
from armbench_segmentation.config import CONFIG
from armbench_segmentation.visualizers.visualizers_getters import multiple_mask_pred, multiple_mask_gt


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

    if containing == 'pred':
        ioas = np.array([[ioa_mask(pred_mask, gt_mask) for gt_mask in gt_masks] for pred_mask in prediction_masks])
    else:
        ioas = np.array([[ioa_mask(gt_mask, pred_mask) for gt_mask in gt_masks] for pred_mask in prediction_masks])

    if len(CACHE_DICTS['get_ioa_array'].keys()) > 2 * CONFIG['BATCH_SIZE']:
        CACHE_DICTS['get_ioa_array'] = {hash_str: ioas}
    else:
        CACHE_DICTS['get_ioa_array'][hash_str] = ioas

    return ioas
