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
from os import makedirs


if __name__ == '__main__':
    model_path = 'model/yolov5.h5'
    if not exists(model_path):
        makedirs(dirname(model_path))
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

    # custom metrics
    general_metric_results = general_metrics_dict(y_true_bbs, y_pred_bbs, y_true_masks, y_pred_masks)
    segmentation_metrics_results = segmentation_metrics_dict(input_img_tf, y_pred_bbs, y_pred_masks, y_true_bbs,
                                                             y_true_masks)
    # visualizers
    #
    gt_mask_visualizer_img = mask_visualizer_gt(image, y_true_bbs[0, ...], y_true_masks[0, ...])
    predicted_mask_visualizer_img = mask_visualizer_prediction(image, y_pred_bbs[0, ...], y_pred_masks[0, ...])
    predicted_bboxes_img = bb_decoder(image, y_pred_bbs[0, ...])
    gt_bboxes_img = gt_bb_decoder(image, y_true_bbs[0, ...])
    under_segmented_img = under_segmented_bb_visualizer(image, y_pred_bbs[0, ...], y_pred_masks[0, ...],
                                                        y_true_bbs[0, ...], y_true_masks[0, ...])
    over_segmented_img = over_segmented_bb_visualizer(image, y_pred_bbs[0, ...], y_pred_masks[0, ...],
                                                      y_true_bbs[0, ...], y_true_masks[0, ...])
    # metadata functions
    for cat in ['tote', 'object']:
        instances = get_cat_instances_seg_lst(idx, training_response, cat)
    metadata = metadata_dict(idx, training_response)
