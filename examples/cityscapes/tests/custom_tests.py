from utils_all.metrics import od_loss
from visualizers.visualizers import gt_bb_decoder
from tensorleap import load_cityscapes_data_leap, metadata_filename, metadata_city, metadata_idx, metadata_gps_heading, \
    metadata_gps_latitude, metadata_gps_longtitude, metadata_outside_temperature, metadata_speed, metadata_yaw_rate, \
    number_of_bb, avg_bb_area_metadata, instances_num, is_class_exist_gen, ground_truth_bbox, \
    input_image, avg_bb_aspect_ratio, label_instances_num, non_normalized_image
import os
# import onnxruntime as ort
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, concatenate
import torch
import torch.nn as nn
from keras.models import load_model
import tensorflow as tf


import onnx
from onnx2keras import onnx_to_keras

def check_custom_integration():
    # preprocess function
    responses = load_cityscapes_data_leap()
    training = responses[0]
    validation = responses[1]
    responses_set = training
    idx = 727
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
    yolo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    y_pred = yolo([concat])
    gt = np.expand_dims(bounding_boxes_gt, axis=0)
    y_true = tf.convert_to_tensor(gt)
    ls = od_loss(y_true, y_pred)
    # conf = confusion_matrix_metric(y_true, y_pred_concat)
    # b = bb_decoder(concat[0], y_pred_concat[0, ...])



    #=============================================================================

    #load model----------------------------
    # path = "/Users/chenrothschild/repo/tensorleap/examples/cityscapes/tests"
    # os.chdir(path)
    # model = os.path.join(path, 'yolov7-tiny_D.onnx')
    #
    # onnx_model = onnx.load(model)  # load onnx model
    # onnx.checker.check_model(onnx_model)  # check onnx model
    # #print(onnx.helper.printable_graph(onnx_model.graph))
    # ort_session = ort.InferenceSession(model)
    # outputs = ort_session.get_outputs()
    #
    #
    # #input------------------------------
    # input_all = [_input.name for _input in onnx_model.graph.input]
    # input_initializer = [node.name for node in onnx_model.graph.initializer]
    # input_names = list(set(input_all) - set(input_initializer))
    #
    # image = np.transpose(image, (2, 0, 1))
    # image_batch = np.expand_dims(image, axis=0)
    # image_batch = image_batch.astype(np.float32)
    #
    # #moel output------------------------
    # y_pred = ort_session.run(None, {input_names[0]: image_batch})
    #
    # #reshape------------
    # new_shapes = [(1, 19200, 85), (1, 4800, 85), (1, 1200, 85)]
    # y_pred_reshape = []
    # for i, pred in enumerate(y_pred):
    #     y_pred_reshape.append(np.reshape(pred, new_shapes[i]))
    #
    # # concat along the channel axis (axis=-1)-----------
    # tensor_0 = torch.from_numpy(y_pred_reshape[0])
    # tensor_1 = torch.from_numpy(y_pred_reshape[1])
    # tensor_2 = torch.from_numpy(y_pred_reshape[2])
    # concat = torch.cat([tensor_0, tensor_1, tensor_2], axis=1)
    #
    # #---------check loss---------------------------
    # bounding_boxes_gt = np.array(bounding_boxes_gt)
    # ls = od_loss(bb_gt=bounding_boxes_gt, y_pred=concat)

if __name__ == '__main__':
    check_custom_integration()
    #check_model(image, bounding_boxes_gt)



