import numpy as np
import tensorflow as tf
from code_loader.contract.visualizer_classes import LeapImageWithBBox, LeapImageMask

from armbench_segmentation.config import CONFIG
from armbench_segmentation.utils.general_utils import get_mask_list, remove_label_from_bbs, \
    get_argmax_map_and_separate_masks
from armbench_segmentation.utils.ioa_utils import get_ioa_array


# Visualizers
def bb_decoder(image, bb_prediction):
    """
    Overlays the BB predictions on the image
    """
    bb_object, _ = get_mask_list(bb_prediction, None, is_gt=False)
    return LeapImageWithBBox((image * 255).astype(np.float32), bb_object)


def gt_bb_decoder(image, bb_gt) -> LeapImageWithBBox:
    bb_object, _ = get_mask_list(bb_gt, None, is_gt=True)
    return LeapImageWithBBox((image * 255).astype(np.float32), bb_object)


def get_mask_visualizer(image, bbs, masks):
    image_size = image.shape[:2]
    argmax_map = np.zeros(image_size, dtype=np.uint8)
    curr_idx = 1
    cats_dict = {}
    cats = []
    for bb, mask in zip(bbs, masks):
        if mask.shape != image_size:
            resize_mask = tf.image.resize(mask[..., None], image_size, tf.image.ResizeMethod.NEAREST_NEIGHBOR)[..., 0]
            if not isinstance(resize_mask, np.ndarray):
                resize_mask = resize_mask.numpy()
        else:
            resize_mask = mask
        resize_mask = resize_mask.astype(bool)
        label = bb.label
        argmax_map[resize_mask] = curr_idx
        instance_number = cats_dict.get(label, 0)
        cats_dict[label] = instance_number + 1
        cats += [f"{label}_{str(instance_number)}"]
        curr_idx += 1
    argmax_map[argmax_map == 0] = curr_idx
    argmax_map -= 1
    return LeapImageMask(mask=argmax_map.astype(np.uint8), image=image.astype(np.float32), labels=cats + ["background"])


def get_mask_visualizer_fixed_instances(image, bbs, masks):
    argmax_map, separate_masks = get_argmax_map_and_separate_masks(image, bbs, masks).values()
    return LeapImageMask(mask=argmax_map.astype(np.uint8), image=image.astype(np.float32),
                         labels=CONFIG['INSTANCES'] + ["background"]), separate_masks


def under_segmented_bb_visualizer(image, y_pred_bb, y_pred_mask, bb_gt, mask_gt):  # bb_visualizer + gt_visualizer
    th = 0.8
    rel_bbs = []
    ioas = get_ioa_array(image, y_pred_bb, y_pred_mask, bb_gt, mask_gt, containing='pred')
    th_arr = ioas > th
    matches_count = th_arr.astype(int).sum(axis=-1)
    relevant_bbs = np.argwhere(matches_count > 1)[..., 0]  # [Indices of bbs]
    relevant_gts = np.where(np.any((th_arr)[relevant_bbs], axis=0))[0]  # [Indices of gts]
    bb_pred_object, _ = get_mask_list(y_pred_bb, None, is_gt=False)
    new_bb_pred_object = remove_label_from_bbs(bb_pred_object, "Tote", "pred")
    bb_gt_object, _ = get_mask_list(bb_gt, None, is_gt=True)
    new_gt_objects = remove_label_from_bbs(bb_gt_object, "Tote", "gt")
    new_bb_array = [new_gt_objects[i] for i in relevant_gts] + [new_bb_pred_object[i] for i in relevant_bbs]
    return LeapImageWithBBox((image * 255).astype(np.float32), new_bb_array)


def over_segmented_bb_visualizer(image, y_pred_bb, y_pred_mask, bb_gt, mask_gt):  # bb_visualizer + gt_visualizer
    th = 0.8
    rel_bbs = []
    ioas = get_ioa_array(image, y_pred_bb, y_pred_mask, bb_gt, mask_gt, containing='gt')
    th_arr = ioas > th
    matches_count = th_arr.astype(int).sum(axis=0)
    relevant_gts = np.argwhere(matches_count > 1)[..., 0]  # [Indices of bbs]
    relevant_bbs = np.where(np.any(th_arr[..., relevant_gts], axis=1))[0]  # [Indices of gts]
    bb_pred_object, _ = get_mask_list(y_pred_bb, None, is_gt=False)
    new_bb_pred_object = remove_label_from_bbs(bb_pred_object, "Tote", "pred")
    bb_gt_object, _ = get_mask_list(bb_gt, None, is_gt=True)
    new_gt_objects = remove_label_from_bbs(bb_gt_object, "Tote", "gt")
    new_bb_array = [new_gt_objects[i] for i in relevant_gts] + [new_bb_pred_object[i] for i in relevant_bbs]
    return LeapImageWithBBox((image * 255).astype(np.float32), new_bb_array)
