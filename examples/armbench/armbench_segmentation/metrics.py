from typing import Tuple, List, Union

import numpy as np
import tensorflow as tf
from code_loader.helpers.detection.yolo.utils import reshape_output_list

from armbench_segmentation.config import CONFIG
from armbench_segmentation.utils.general_utils import remove_label_from_bbs
from armbench_segmentation.yolo_helpers.yolo_utils import LOSS_FN
from code_loader.contract.responsedataclasses import BoundingBox


def compute_losses(obj_true: tf.Tensor, od_pred: tf.Tensor,
                   mask_true, instance_seg: tf.Tensor) -> Union[
    Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]],
    Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]]]:
    """
    Computes the sum of the classification (CE loss) and localization (regression) losses from all heads
    """
    decoded = False if CONFIG["MODEL_FORMAT"] != "inference" else True
    class_list_reshaped, loc_list_reshaped = reshape_output_list(od_pred, decoded=decoded,
                                                                 image_size=CONFIG["IMAGE_SIZE"])  # add batch
    loss_l, loss_c, loss_o, loss_m = LOSS_FN(y_true=obj_true, y_pred=(loc_list_reshaped, class_list_reshaped),
                                             instance_seg=instance_seg, instance_true=mask_true)
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


def over_under_segmented_metrics(batched_ioas_list: List[np.ndarray], count_small_bbs=False, get_avg_confidence=False,
                                 bb_mask_object_list: List[Union[List[BoundingBox], List[np.ndarray]]] = None):
    th = 0.8
    segmented_arr = [0.]*len(batched_ioas_list)
    segmented_arr_count = [0.]*len(batched_ioas_list)
    average_segments_amount = [0.]*len(batched_ioas_list)
    conf_arr = [0.]*len(batched_ioas_list)
    has_small_bbs = [0.]*len(batched_ioas_list)
    for batch in range(len(batched_ioas_list)):
        ioas = batched_ioas_list[batch]
        if len(ioas) > 0:
            th_arr = ioas > th
            matches_count = th_arr.astype(int).sum(axis=-1)
            is_over_under_segmented = float(len(matches_count[matches_count > 1]) > 0)
            over_under_segmented_count = float(len(matches_count[matches_count > 1]))
            if over_under_segmented_count > 0:
                average_segments_num_over_under = float(matches_count[matches_count > 1].mean())
            else:
                average_segments_num_over_under = 0.
            average_segments_amount[batch] = average_segments_num_over_under
            segmented_arr_count[batch] = over_under_segmented_count
            segmented_arr[batch] = is_over_under_segmented
            if count_small_bbs or get_avg_confidence:
                relevant_bbs = np.argwhere(matches_count > 1)[..., 0]  # [Indices of bbs]
                relevant_gts = np.where(np.any(th_arr[relevant_bbs], axis=0))[0]  # [Indices of gts]
                if count_small_bbs:
                    new_gt_objects = remove_label_from_bbs(bb_mask_object_list[batch][0], "Tote", "gt")
                    new_bb_array = [new_gt_objects[i] for i in relevant_gts]
                    for j in range(len(new_bb_array)):
                        if new_bb_array[j].width * new_bb_array[j].height < CONFIG["SMALL_BBS_TH"]:
                            has_small_bbs[batch] = 1.
                if get_avg_confidence:
                    new_bb_pred_object = remove_label_from_bbs(bb_mask_object_list[batch][0], "Tote", "pred")
                    new_bb_array = [new_bb_pred_object[j] for j in relevant_gts]
                    if len(new_bb_array) > 0:
                        avg_conf = np.array([new_bb_array[j].confidence for j in range(len(new_bb_array))]).mean()
                    else:
                        avg_conf = 0.
                    conf_arr[batch] = avg_conf
    return tf.convert_to_tensor(segmented_arr), tf.convert_to_tensor(segmented_arr_count),\
           tf.convert_to_tensor(average_segments_amount), tf.convert_to_tensor(has_small_bbs),\
           tf.convert_to_tensor(conf_arr)
