import urllib
from os.path import exists

import numpy as np
import tensorflow as tf

from utils_all.general_utils import get_json, get_polygon
from utils_all.metrics import od_loss, object_metric, classification_metric, regression_metric
from utils_all.plots import plot_image_with_polygons, plot_image_with_bboxes
from utils_all.preprocessing import CATEGORIES_no_background, CATEGORIES_id_no_background
from visualizers.visualizers import gt_bb_decoder, bb_decoder, bb_car_decoder, bb_car_gt_decoder

from leap_binder import metadata_dict, load_cityscapes_data_leap, metadata_filename, metadata_city, metadata_idx, \
    metadata_gps_heading, \
    metadata_gps_latitude, metadata_gps_longtitude, metadata_outside_temperature, metadata_speed, metadata_yaw_rate, \
    number_of_bb, avg_bb_area_metadata, instances_num, is_class_exist_gen, ground_truth_bbox, \
    avg_bb_aspect_ratio, label_instances_num, non_normalized_image, count_small_bbs, is_class_exist_veg_and_building, \
    metadata_brightness, metadata_person_category_avg_size, metadata_car_category_avg_size, get_class_mean_iou, \
    od_metrics_dict


def check_custom_integration():
    # preprocess function
    responses = load_cityscapes_data_leap()
    train = responses[0]
    val = responses[1]
    test = responses[2]
    responses_set = val
    for idx in range(20):
        # get input and gt
        image = non_normalized_image(idx, responses_set)
        bounding_boxes_gt = ground_truth_bbox(idx, responses_set)

        json_data = get_json(idx, responses_set)
        image_height, image_width = json_data['imgHeight'], json_data['imgWidth']
        polygons = get_polygon(json_data)
        plot_image_with_polygons(image_height, image_width, polygons, image)

        # model
        if not exists('yolov7.h5'):
            print("Downloading yolov7 for inference")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/example-datasets-47ml982d/yolov7/yolov7.h5", "yolov7.h5")
        yolo = tf.keras.models.load_model("yolov7.h5")
        concat = np.expand_dims(image, axis=0)
        y_pred = yolo([concat])
        gt = np.expand_dims(bounding_boxes_gt, axis=0)
        y_true = tf.convert_to_tensor(gt)

        # get visualizer
        bb_gt_decoder = gt_bb_decoder(image, y_true)
        plot_image_with_bboxes(image, bb_gt_decoder.bounding_boxes, 'gt')
        bb__decoder = bb_decoder(image, y_pred[0, ...])
        plot_image_with_bboxes(image, bb__decoder.bounding_boxes, 'pred')
        bb_car = bb_car_decoder(image, y_pred[0, ...])
        plot_image_with_bboxes(image, bb_car.bounding_boxes, 'pred')
        bb_gt_car = bb_car_gt_decoder(image, y_true)
        plot_image_with_bboxes(image, bb_gt_car.bounding_boxes, 'gt')

        # get custom metrics
        ls = od_loss(y_true, y_pred)
        metadata_all = metadata_dict(idx, responses_set)
        metrices_all = od_metrics_dict(y_true, y_pred)

        # Regression_metric = regression_metric(y_true, y_pred)
        # Classification_metric = classification_metric(y_true, y_pred)
        # Object_metric = object_metric(y_true, y_pred)
        # print(f"regression loss is {tf.stack(Regression_metric).numpy()}")
        # print(f"classification loss is {tf.stack(Classification_metric).numpy()}")
        # print(f"object loss is {tf.stack(Object_metric).numpy()}")



        # # get meata_data
        # for class_id in CATEGORIES_id_no_background:
        #     iou_id = get_class_mean_iou(class_id)
        #     iou_id_ = iou_id(y_true, y_pred)
        #     print(f"iou_id of {class_id} is {tf.stack(iou_id_).numpy()}")

        # brightness = metadata_brightness(idx, responses_set)
        # person_category_avg = metadata_person_category_avg_size(idx, responses_set)
        # print(f'person_category_avg: {person_category_avg}')
        # car_category_avg = metadata_car_category_avg_size(idx, responses_set)
        # print(f'car_category_avg: {car_category_avg}')
        # file_name = metadata_filename(idx, responses_set)
        # city = metadata_city(idx, responses_set)
        # idx = metadata_idx(idx, responses_set)
        # gps_heading = metadata_gps_heading(idx, responses_set)
        # gps_latitude = metadata_gps_latitude(idx, responses_set)
        # gps_longtitude = metadata_gps_longtitude(idx, responses_set)
        # outside_temperature = metadata_outside_temperature(idx, responses_set)
        # speed = metadata_speed(idx, responses_set)
        # yaw_rate = metadata_yaw_rate(idx, responses_set)
        # bb_count = number_of_bb(idx, responses_set)
        # Avg_bb_aspect_ratio = avg_bb_aspect_ratio(idx, responses_set)
        # avg_bb_area = avg_bb_area_metadata(idx, responses_set)
        # instances_number_metadata = instances_num(idx, responses_set)
        # small_bbs_number = count_small_bbs(idx, responses_set)
        # for class_label in CATEGORIES_no_background:
        #     instances_count_func = label_instances_num(class_label)
        #     instances_count = instances_count_func(idx, responses_set)
        # for class_id in CATEGORIES_id_no_background:
        #     class_exist_func = is_class_exist_gen(class_id)
        #     class_exist = class_exist_func(idx, responses_set)
        # vb_class_exist_func = is_class_exist_veg_and_building(21, 11)
        # class_exist = vb_class_exist_func(idx, responses_set)

if __name__ == '__main__':
    check_custom_integration()



