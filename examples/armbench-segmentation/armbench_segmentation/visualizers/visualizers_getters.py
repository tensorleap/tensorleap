from armbench_segmentation.utils.general_utils import get_mask_list, get_argmax_map_and_separate_masks


def mask_visualizer_gt(image, bb_gt, masks_gt):
    bbs, masks = get_mask_list(bb_gt, masks_gt, is_gt=True)
    return get_argmax_map_and_separate_masks(image, bbs, masks)["argmax_map"]


def mask_visualizer_prediction(image, y_pred_bbs, y_pred_mask):
    bbs, masks = get_mask_list(y_pred_bbs, y_pred_mask, is_gt=False)
    return get_argmax_map_and_separate_masks(image, bbs, masks)["argmax_map"]


def multiple_mask_gt(image, data, mask):
    bbs, masks = get_mask_list(data, mask, is_gt=True)
    return get_argmax_map_and_separate_masks(image, bbs, masks)['separate_masks']


def multiple_mask_pred(image, data, mask):
    bbs, masks = get_mask_list(data, mask, is_gt=False)
    return get_argmax_map_and_separate_masks(image, bbs, masks)['separate_masks']