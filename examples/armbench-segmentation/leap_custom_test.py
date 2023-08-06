import urllib
from os.path import exists

import tensorflow as tf
import numpy as np

from armbench_segmentation.metrics import (
    regression_metric, classification_metric, object_metric, mask_metric, over_segmented, under_segmented,
    metric_small_bb_in_under_segment, over_segmented_instances_count, under_segmented_instances_count,
    average_segments_num_over_segment, average_segments_num_under_segmented, over_segment_avg_confidence
)
from armbench_segmentation.visualizers.visualizers import (
    gt_bb_decoder, bb_decoder, under_segmented_bb_visualizer, over_segmented_bb_visualizer
)
from armbench_segmentation.visualizers.visualizers_getters import mask_visualizer_gt, mask_visualizer_prediction
from leap_binder import (
    subset_images, input_image, get_bbs, get_masks, get_cat_instances_seg_lst, get_idx, get_fname, get_original_width,
    get_original_height, bbox_num, get_avg_bb_area, get_avg_bb_aspect_ratio, get_instances_num,
    get_object_instances_num, get_tote_instances_num, get_avg_instance_percent, get_tote_instances_masks,
    get_tote_instances_sizes, get_tote_avg_instance_percent, get_tote_std_instance_percent, get_tote_instances_mean,
    get_tote_instances_std, get_object_instances_mean, get_object_instances_std, get_object_instances_masks,
    get_object_instances_sizes, get_object_avg_instance_percent, get_object_std_instance_percent,
    get_background_percent, get_obj_bbox_occlusions_count, get_obj_bbox_occlusions_avg, get_obj_mask_occlusions_count,
    count_duplicate_bbs, count_small_bbs
)

if __name__ == '__main__':
    model_path = 'model/yolov5.h5'
    if not exists(model_path):
        print("Downloading YOLOv5.h5 for inference")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/example-datasets-47ml982d/yolov5/yolov5.h5",
            model_path)
    model = tf.keras.models.load_model(model_path)
    idx = 0
    responses = subset_images()  # get dataset splits
    training_response = responses[0]  # [training, validation, test]
    image = input_image(idx, training_response)  # get specific image
    bb_gt = get_bbs(idx, training_response)  # get bounding boxes for the image
    mask_gt = get_masks(idx, training_response)  # get the corresponding segmentation masks
    y_true_bbs = tf.expand_dims(tf.convert_to_tensor(bb_gt), 0)  # convert ground truth bbs to tensor
    y_true_masks = tf.expand_dims(tf.convert_to_tensor(mask_gt), 0)  # convert ground truth bbs to tensor

    input_img = np.expand_dims(image, axis=0)  # prepare input for YOLOv5
    input_img_tf = tf.convert_to_tensor(input_img)
    y_pred = model([input_img])  # infer and get model prediction
    y_pred_bbs = y_pred[0]
    y_pred_masks = y_pred[1]
    y_pred_bb_concat = tf.keras.layers.Permute((2, 1))(y_pred_bbs)  # prepare prediction for further use

    # visualizers
    gt_mask_visualizer_img = mask_visualizer_gt(image, bb_gt, mask_gt)
    predicted_mask_visualizer_img = mask_visualizer_prediction(image, y_pred_bbs[0, ...], y_pred_masks[0, ...])
    predicted_bboxes_img = bb_decoder(image, y_pred_bbs[0, ...])
    gt_bboxes_img = gt_bb_decoder(image, bb_gt)
    under_segmented_img = under_segmented_bb_visualizer(image, y_pred_bbs[0, ...], y_pred_masks[0, ...], bb_gt, mask_gt)
    over_segmented_img = over_segmented_bb_visualizer(image, y_pred_bbs[0, ...], y_pred_masks[0, ...], bb_gt, mask_gt)

    # custom metrics
    regression_metric_result = regression_metric(y_true_bbs, y_pred_bbs, y_true_masks, y_pred_masks)
    classification_metric_result = classification_metric(y_true_bbs, y_pred_bbs, y_true_masks, y_pred_masks)
    object_metric_result = object_metric(y_true_bbs, y_pred_bbs, y_true_masks, y_pred_masks)
    mask_metric_results = mask_metric(y_true_bbs, y_pred_bbs, y_true_masks, y_pred_masks)
    over_segmented_result = over_segmented(input_img_tf, y_pred_bbs, y_pred_masks, y_true_bbs, y_true_masks)
    under_segmented_result = under_segmented(input_img_tf, y_pred_bbs, y_pred_masks, y_true_bbs, y_true_masks)
    metric_small_bb_in_under_segment_result = metric_small_bb_in_under_segment(input_img_tf, y_pred_bbs, y_pred_masks,
                                                                               y_true_bbs, y_true_masks)
    over_segmented_instances_count_result = over_segmented_instances_count(input_img_tf, y_pred_bbs, y_pred_masks,
                                                                           y_true_bbs, y_true_masks)
    under_segmented_instances_count_result = under_segmented_instances_count(input_img_tf, y_pred_bbs, y_pred_masks,
                                                                             y_true_bbs, y_true_masks)
    average_segments_num_over_segment_result = average_segments_num_over_segment(input_img_tf, y_pred_bbs, y_pred_masks,
                                                                                 y_true_bbs, y_true_masks)
    average_segments_num_under_segmented_result = average_segments_num_under_segmented(input_img_tf, y_pred_bbs,
                                                                                       y_pred_masks, y_true_bbs,
                                                                                       y_true_masks)
    over_segment_avg_confidence_result = over_segment_avg_confidence(input_img_tf, y_pred_bbs, y_pred_masks, y_true_bbs,
                                                                     y_true_masks)

    # metadata functions
    for cat in ['tote', 'object']:
        instances = get_cat_instances_seg_lst(idx, training_response, cat)
    index = get_idx(idx, training_response)
    filename = get_fname(idx, training_response)
    original_width = get_original_width(idx, training_response)
    original_height = get_original_height(idx, training_response)
    n_bboxes = bbox_num(idx, training_response)
    avg_bbox_area = get_avg_bb_area(idx, training_response)
    avg_bbox_aspect_ratio = get_avg_bb_aspect_ratio(idx, training_response)
    n_instances = get_instances_num(idx, training_response)
    n_objects = get_object_instances_num(idx, training_response)
    n_tote = get_tote_instances_num(idx, training_response)
    avg_instance_percent = get_avg_instance_percent(idx, training_response)
    tote_masks = get_tote_instances_masks(idx, training_response)
    tote_sizes = get_tote_instances_sizes(idx, training_response)
    tote_avg_instance_percent = get_tote_avg_instance_percent(idx, training_response)
    tote_std_instance_percent = get_tote_std_instance_percent(idx, training_response)
    tote_instances_mean = get_tote_instances_mean(idx, training_response)
    tote_instances_std = get_tote_instances_std(idx, training_response)
    object_instances_mean = get_object_instances_mean(idx, training_response)
    object_instances_std = get_object_instances_std(idx, training_response)
    object_instances_masks = get_object_instances_masks(idx, training_response)
    object_instances_sizes = get_object_instances_sizes(idx, training_response)
    object_avg_instance_percent = get_object_avg_instance_percent(idx, training_response)
    object_std_instance_percent = get_object_std_instance_percent(idx, training_response)
    background_percent = get_background_percent(idx, training_response)
    obj_bbox_occlusions_count = get_obj_bbox_occlusions_count(idx, training_response)
    obj_bbox_occlusions_avg = get_obj_bbox_occlusions_avg(idx, training_response)
    obj_mask_occlusions_count = get_obj_mask_occlusions_count(idx, training_response)
    duplicate_bb = count_duplicate_bbs(idx, training_response)
    small_bbs = count_small_bbs(idx, training_response)
