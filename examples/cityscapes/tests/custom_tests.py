from utils_all.metrics import od_loss
from visualizers.visualizers import gt_bb_decoder
from tensorleap import load_cityscapes_data_leap, metadata_filename, metadata_city, metadata_idx, metadata_gps_heading, \
    metadata_gps_latitude, metadata_gps_longtitude, metadata_outside_temperature, metadata_speed, metadata_yaw_rate, \
    number_of_bb, avg_bb_area_metadata, instances_num, is_class_exist_gen, ground_truth_bbox, \
    input_image, avg_bb_aspect_ratio, label_instances_num, non_normalized_image
import onnx
import os
# import onnxruntime as ort
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, concatenate
import torch
import torch.nn as nn
from keras.models import load_model
import tensorflow as tf

def check_custom_integration():
    # preprocess function
    responses = load_cityscapes_data_leap()
    training = responses[0]
    validation = responses[1]
    responses_set = training
    idx = 899
    # set input and gt
    image = non_normalized_image(idx, responses_set)
    bounding_boxes_gt = ground_truth_bbox(idx, responses_set)

    check_model(image, bounding_boxes_gt)


    # set meata_data
    file_name = metadata_filename(idx, responses_set)
    city = metadata_city(idx, responses_set)
    idx = metadata_idx(idx, responses_set)
    gps_heading = metadata_gps_heading(idx, responses_set)
    gps_latitude = metadata_gps_latitude(idx, responses_set)
    gps_longtitude = metadata_gps_longtitude(idx, responses_set)
    outside_temperature = metadata_outside_temperature(idx, responses_set)
    speed = metadata_speed(idx, responses_set)
    yaw_rate = metadata_yaw_rate(idx, responses_set)
    bb_count = number_of_bb(idx, responses_set)
    Avg_bb_aspect_ratio = avg_bb_aspect_ratio(idx, responses_set)
    avg_bb_area = avg_bb_area_metadata(idx, responses_set)
    instances_number_metadata = instances_num(idx, responses_set)
    # small_bbs_number = count_small_bbs(idx, responses_set)

    # does_0_exist = is_class_exist_gen(idx, responses_set)  # TODO: check
    # Label_instances_num = label_instances_num(idx, responses_set)  # TODO: check

    # set visualizer
    bb_gt_decoder = gt_bb_decoder(image, bounding_boxes_gt)
    # bb_decoder = bb_decoder(image, detection_pred) #TODO: check

    # set custom metrics
    # Regression_metric = regression_metric(bounding_boxes_gt, detection_pred)#TODO: check
    # Classification_metric = classification_metric(bounding_boxes_gt, detection_pred)#TODO: check
    # Object_metric = object_metric(bounding_boxes_gt, detection_pred)#TODO: check

def check_model(image, bounding_boxes_gt):
    #------------export model----------------------------
    path = "/Users/chenrothschild/repo/tensorleap/examples/cityscapes/model"
    os.chdir(path)
    model = os.path.join(path, 'exported-model.h5')
    yolo = tf.keras.models.load_model(model)

    concat = np.expand_dims(image, axis=0)
    #concat = np.concatenate([concat, concat], axis=0)
    y_pred = yolo([concat])
    #y_pred_concat = tf.keras.layers.Permute((2, 1))(y_pred)
    gt = np.expand_dims(bounding_boxes_gt, axis=0)
    y_true = tf.convert_to_tensor(gt)
    ls = od_loss(y_true, y_pred)
    # conf = confusion_matrix_metric(y_true, y_pred_concat)
    # b = bb_decoder(concat[0], y_pred_concat[0, ...])





if __name__ == '__main__':
    check_custom_integration()
    #check_model(image, bounding_boxes_gt)



