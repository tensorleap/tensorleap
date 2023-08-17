import os
import urllib
from os.path import exists, dirname

import tensorflow as tf
import numpy as np

from armbench_segmentation.visualizers.visualizers import (
    gt_bb_decoder, bb_decoder, under_segmented_bb_visualizer, over_segmented_bb_visualizer
)
from armbench_segmentation.visualizers.visualizers_getters import mask_visualizer_gt, mask_visualizer_prediction
from leap_binder import (
    subset_images, input_image, get_bbs, get_masks, get_cat_instances_seg_lst, general_metrics_dict,
    segmentation_metrics_dict, metadata_dict
)

def check_integration():
    model_path = 'model/yolov5.h5'
    if not exists(model_path):
        os.makedirs('model', exist_ok=True)
        print("Downloading YOLOv5.h5 for inference")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/example-datasets-47ml982d/yolov5/yolov5.h5",
            model_path)
    model = tf.keras.models.load_model(model_path)
    batch = 64
    responses = subset_images()  # get dataset splits
    training_response = responses[0]  # [training, validation, test]
    images = []
    bb_gt = []
    mask_gt =[]
    for idx in range(batch):
        images.append(input_image(idx, training_response))
        bb_gt.append(get_bbs(idx, training_response))
        mask_gt.append(get_masks(idx, training_response))
    y_true_bbs = tf.convert_to_tensor(bb_gt) # convert ground truth bbs to tensor
    y_true_masks = tf.convert_to_tensor(mask_gt)  # convert ground truth bbs to tensor

    input_img_tf = tf.convert_to_tensor(images)
    y_pred = model([input_img_tf])  # infer and get model prediction
    y_pred_bbs = y_pred[0]
    y_pred_masks = y_pred[1]
    y_pred_bb_concat = tf.keras.layers.Permute((2, 1))(y_pred_bbs)  # prepare prediction for further use

    # custom metrics
    general_metric_results = general_metrics_dict(y_true_bbs, y_pred_bbs, y_true_masks, y_pred_masks)
    segmentation_metrics_results = segmentation_metrics_dict(input_img_tf, y_pred_bbs, y_pred_masks, y_true_bbs,
                                                             y_true_masks)
    # visualizers
    #
    gt_mask_visualizer_img = mask_visualizer_gt(images[0], y_true_bbs[0, ...], y_true_masks[0, ...])
    predicted_mask_visualizer_img = mask_visualizer_prediction(images[0], y_pred_bbs[0, ...], y_pred_masks[0, ...])
    predicted_bboxes_img = bb_decoder(images[0], y_pred_bbs[0, ...])
    gt_bboxes_img = gt_bb_decoder(images[0], y_true_bbs[0, ...])
    under_segmented_img = under_segmented_bb_visualizer(images[0], y_pred_bbs[0, ...], y_pred_masks[0, ...],
                                                        y_true_bbs[0, ...], y_true_masks[0, ...])
    over_segmented_img = over_segmented_bb_visualizer(images[0], y_pred_bbs[0, ...], y_pred_masks[0, ...],
                                                      y_true_bbs[0, ...], y_true_masks[0, ...])
    # metadata functions
    for cat in ['tote', 'object']:
        instances = get_cat_instances_seg_lst(idx, training_response, cat)
    metadata = metadata_dict(idx, training_response)


if __name__ == '__main__':
    check_integration()
