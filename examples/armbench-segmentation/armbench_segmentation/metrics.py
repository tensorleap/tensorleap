from typing import Tuple, List, Union

import numpy as np
import tensorflow as tf
from code_loader.helpers.detection.yolo.utils import reshape_output_list

from armbench_segmentation import CACHE_DICTS
from armbench_segmentation.config import CONFIG
from armbench_segmentation.utils.general_utils import get_mask_list, remove_label_from_bbs
from armbench_segmentation.utils.ioa_utils import get_ioa_array
from armbench_segmentation.yolo_helpers.yolo_utils import LOSS_FN


def compute_losses(obj_true: tf.Tensor, od_pred: tf.Tensor,
                   mask_true, instance_seg: tf.Tensor) -> Union[
    Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]],
    Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]]]:
    """
    Computes the sum of the classification (CE loss) and localization (regression) losses from all heads
    """
    res = CACHE_DICTS['loss'].get(str(obj_true) + str(od_pred) + str(mask_true) + str(instance_seg))
    if res is not None:
        return res
    decoded = False if CONFIG["MODEL_FORMAT"] != "inference" else True
    class_list_reshaped, loc_list_reshaped = reshape_output_list(od_pred, decoded=decoded,
                                                                 image_size=CONFIG["IMAGE_SIZE"])  # add batch
    loss_l, loss_c, loss_o, loss_m = LOSS_FN(y_true=obj_true, y_pred=(loc_list_reshaped, class_list_reshaped),
                                             instance_seg=instance_seg, instance_true=mask_true)
    CACHE_DICTS['loss'] = {
        str(obj_true) + str(od_pred) + str(mask_true) + str(instance_seg): (loss_l, loss_c, loss_o, loss_m)}
    return loss_l, loss_c, loss_o, loss_m


def instance_seg_loss(bb_gt: tf.Tensor, detection_pred: tf.Tensor,
                      mask_gt: tf.Tensor, segmentation_pred: tf.Tensor):  # return batch
    """
    Sums the classification and regression loss
    """
    loss_l, loss_c, loss_o, loss_m = compute_losses(bb_gt, detection_pred, mask_gt, segmentation_pred)
    combined_losses = [l + c + o + m for l, c, o, m in zip(loss_l, loss_c, loss_o, loss_m)]
    sum_loss = tf.reduce_sum(combined_losses, axis=0)
    return sum_loss


def classification_metric(bb_gt: tf.Tensor, detection_pred: tf.Tensor,
                          mask_gt: tf.Tensor, segmentation_pred: tf.Tensor):  # return batch
    _, loss_c, _, _ = compute_losses(bb_gt, detection_pred, mask_gt, segmentation_pred)
    return tf.reduce_sum(loss_c, axis=0)[:, 0]


def regression_metric(bb_gt: tf.Tensor, detection_pred: tf.Tensor,
                      mask_gt: tf.Tensor, segmentation_pred: tf.Tensor):  # return batch
    loss_l, _, _, _ = compute_losses(bb_gt, detection_pred, mask_gt, segmentation_pred)
    return tf.reduce_sum(loss_l, axis=0)[:, 0]  # shape of batch


def object_metric(bb_gt: tf.Tensor, detection_pred: tf.Tensor,
                  mask_gt: tf.Tensor, segmentation_pred: tf.Tensor):
    _, _, loss_o, _ = compute_losses(bb_gt, detection_pred, mask_gt, segmentation_pred)
    return tf.reduce_sum(loss_o, axis=0)[:, 0]  # shape of batch


def mask_metric(bb_gt: tf.Tensor, detection_pred: tf.Tensor,
                mask_gt: tf.Tensor, segmentation_pred: tf.Tensor):
    _, _, _, loss_m = compute_losses(bb_gt, detection_pred, mask_gt, segmentation_pred)
    return tf.reduce_sum(loss_m, axis=0)[:, 0]  # shape of batch


def under_segmented(image: tf.Tensor, y_pred_bb: tf.Tensor, y_pred_mask: tf.Tensor, bb_gt: tf.Tensor,
                    mask_gt: tf.Tensor):
    th = 0.8
    under_segmented_arr = []
    for i in range(image.shape[0]):
        ioas = get_ioa_array(image[i, ...], y_pred_bb[i, ...], y_pred_mask[i, ...], bb_gt[i, ...], mask_gt[i, ...],
                             containing='pred')
        matches_count = (ioas > th).astype(int).sum(axis=-1)
        is_under_segmented = float(len(matches_count[matches_count > 1]) > 0)
        under_segmented_arr.append(is_under_segmented)
    return tf.convert_to_tensor(under_segmented_arr)


def over_segmented(image: tf.Tensor, y_pred_bb: tf.Tensor, y_pred_mask: tf.Tensor, bb_gt: tf.Tensor,
                   mask_gt: tf.Tensor):
    th = 0.8
    over_segmented_arr = []
    for i in range(image.shape[0]):
        ioas = get_ioa_array(image[i, ...], y_pred_bb[i, ...], y_pred_mask[i, ...], bb_gt[i, ...], mask_gt[i, ...],
                             containing='gt')
        matches_count = (ioas > th).astype(int).sum(axis=0)
        is_over_segmented = float(len(matches_count[matches_count > 1]) > 0)
        over_segmented_arr.append(is_over_segmented)
    return tf.convert_to_tensor(over_segmented_arr)


def metric_small_bb_in_under_segment(image: tf.Tensor, y_pred_bb: tf.Tensor, y_pred_mask: tf.Tensor, bb_gt: tf.Tensor,
                                     mask_gt: tf.Tensor):  # bb_visualizer + gt_visualizer
    th = 0.8  # equivelan
    has_small_bbs = [0.] * image.shape[0]
    for i in range(image.shape[0]):
        ioas = get_ioa_array(image[i, ...], y_pred_bb[i, ...], y_pred_mask[i, ...], bb_gt[i, ...], mask_gt[i, ...],
                             containing='pred')
        th_arr = ioas > th
        matches_count = th_arr.astype(int).sum(axis=-1)
        relevant_bbs = np.argwhere(matches_count > 1)[..., 0]  # [Indices of bbs]
        relevant_gts = np.where(np.any((th_arr)[relevant_bbs], axis=0))[0]  # [Indices of gts]
        bb_gt_object, _ = get_mask_list(bb_gt, None, is_gt=True)
        new_gt_objects = remove_label_from_bbs(bb_gt_object, "Tote", "gt")
        new_bb_array = [new_gt_objects[i] for i in relevant_gts]
        for j in range(len(new_bb_array)):
            if new_bb_array[j].width * new_bb_array[j].height < CONFIG["SMALL_BBS_TH"]:
                has_small_bbs[i] = 1.
    return tf.convert_to_tensor(has_small_bbs)


def non_binary_over_segmented(image: tf.Tensor, y_pred_bb: tf.Tensor, y_pred_mask: tf.Tensor, bb_gt: tf.Tensor,
                              mask_gt: tf.Tensor):
    th = 0.8
    over_segmented_arr = []
    for i in range(image.shape[0]):
        ioas = get_ioa_array(image[i, ...], y_pred_bb[i, ...], y_pred_mask[i, ...], bb_gt[i, ...], mask_gt[i, ...],
                             containing='gt')
        matches_count = (ioas > th).astype(int).sum(axis=0)
        is_over_segmented = float(len(matches_count[matches_count > 1]))
        over_segmented_arr.append(is_over_segmented)
    return tf.convert_to_tensor(over_segmented_arr)


def non_binary_under_segmented(image: tf.Tensor, y_pred_bb: tf.Tensor, y_pred_mask: tf.Tensor, bb_gt: tf.Tensor,
                               mask_gt: tf.Tensor):
    th = 0.8
    under_segmented_arr = []
    for i in range(image.shape[0]):
        ioas = get_ioa_array(image[i, ...], y_pred_bb[i, ...], y_pred_mask[i, ...], bb_gt[i, ...], mask_gt[i, ...],
                             containing='pred')
        matches_count = (ioas > th).astype(int).sum(axis=-1)
        is_under_segmented = float(len(matches_count[matches_count > 1]))
        under_segmented_arr.append(is_under_segmented)
    return tf.convert_to_tensor(under_segmented_arr)


def average_segments_num_over_segment(image: tf.Tensor, y_pred_bb: tf.Tensor, y_pred_mask: tf.Tensor, bb_gt: tf.Tensor,
                                      mask_gt: tf.Tensor):
    th = 0.8
    over_segmented_arr = []
    for i in range(image.shape[0]):
        ioas = get_ioa_array(image[i, ...], y_pred_bb[i, ...], y_pred_mask[i, ...], bb_gt[i, ...], mask_gt[i, ...],
                             containing='gt')
        matches_count = (ioas > th).astype(int).sum(axis=0)
        if len(matches_count[matches_count > 1]) > 0:
            is_over_segmented = float(matches_count[matches_count > 1].mean())
        else:
            is_over_segmented = 0.
        over_segmented_arr.append(is_over_segmented)
    return tf.convert_to_tensor(over_segmented_arr)


def average_segments_num_under_segmented(image: tf.Tensor, y_pred_bb: tf.Tensor, y_pred_mask: tf.Tensor,
                                         bb_gt: tf.Tensor, mask_gt: tf.Tensor):
    th = 0.8
    under_segmented_arr = []
    for i in range(image.shape[0]):
        ioas = get_ioa_array(image[i, ...], y_pred_bb[i, ...], y_pred_mask[i, ...], bb_gt[i, ...], mask_gt[i, ...],
                             containing='pred')
        matches_count = (ioas > th).astype(int).sum(axis=-1)
        if len(matches_count[matches_count > 1]) > 0:
            is_under_segmented = float(matches_count[matches_count > 1].mean())
        else:
            is_under_segmented = 0.
        under_segmented_arr.append(is_under_segmented)
    return tf.convert_to_tensor(under_segmented_arr)


def over_segment_avg_confidence(image: tf.Tensor, y_pred_bb: tf.Tensor, y_pred_mask: tf.Tensor, bb_gt: tf.Tensor,
                                mask_gt: tf.Tensor):  # bb_visualizer + gt_visualizer
    th = 0.8
    conf_arr = []
    for i in range(image.shape[0]):
        ioas = get_ioa_array(image[i, ...], y_pred_bb[i, ...], y_pred_mask[i, ...], bb_gt[i, ...], mask_gt[i, ...],
                             containing='gt')
        th_arr = ioas > th
        matches_count = th_arr.astype(int).sum(axis=0)
        relevant_gts = np.argwhere(matches_count > 1)[..., 0]  # [Indices of bbs]
        relevant_bbs = np.where(np.any(th_arr[..., relevant_gts], axis=1))[0]  # [Indices of gts]
        bb_pred_object, _ = get_mask_list(y_pred_bb[i, ...], None, is_gt=False)
        new_bb_pred_object = remove_label_from_bbs(bb_pred_object, "Tote", "pred")
        bb_gt_object, _ = get_mask_list(bb_gt, None, is_gt=True)
        new_bb_array = [new_bb_pred_object[j] for j in relevant_bbs]
        if len(new_bb_array) > 0:
            avg_conf = np.array([new_bb_array[j].confidence for j in range(len(new_bb_array))]).mean()
        else:
            avg_conf = 0.
        conf_arr.append(avg_conf)
    return tf.convert_to_tensor(conf_arr)
