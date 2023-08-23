from armbench_segmentation.utils.general_utils import get_mask_list, get_argmax_map_and_separate_masks
import numpy as np
from armbench_segmentation.config import CONFIG
from code_loader.contract.visualizer_classes import LeapImageMask


def mask_visualizer_gt(image, bb_gt, masks_gt):
    bbs, masks = get_mask_list(bb_gt, masks_gt, is_gt=True)
    argmax_arr = get_argmax_map_and_separate_masks(image, bbs, masks)["argmax_map"].astype(np.uint8)
    return LeapImageMask(mask=argmax_arr.astype(np.uint8), image=image.astype(np.float32),
                         labels=CONFIG['INSTANCES'] + ["background"])


def mask_visualizer_prediction(image, y_pred_bbs, y_pred_mask):
    bbs, masks = get_mask_list(y_pred_bbs, y_pred_mask, is_gt=False)
    argmax_arr = get_argmax_map_and_separate_masks(image, bbs, masks)["argmax_map"].astype(np.uint8)
    return LeapImageMask(mask=argmax_arr.astype(np.uint8), image=image.astype(np.float32),
                         labels=CONFIG['INSTANCES'] + ["background"])


def multiple_mask_gt(image, data, mask):
    bbs, masks = get_mask_list(data, mask, is_gt=True)
    return get_argmax_map_and_separate_masks(image, bbs, masks)['separate_masks']


def multiple_mask_pred(image, data, mask):
    bbs, masks = get_mask_list(data, mask, is_gt=False)
    return get_argmax_map_and_separate_masks(image, bbs, masks)['separate_masks']
