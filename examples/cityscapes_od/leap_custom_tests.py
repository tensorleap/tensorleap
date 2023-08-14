import urllib
from os.path import exists
import numpy as np
import tensorflow as tf

from cityscapes_od.metrics import od_loss
from cityscapes_od.plots import plot_image_with_polygons, plot_image_with_bboxes
from cityscapes_od.utils.general_utils import get_json, get_polygon
from leap_binder import metadata_dict, load_cityscapes_data_leap, ground_truth_bbox, non_normalized_image, \
    od_metrics_dict, gt_bb_decoder, bb_decoder, bb_car_decoder, bb_car_gt_decoder


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

        # get custom metrics and meta data
        ls = od_loss(y_true, y_pred)
        metrices_all = od_metrics_dict(y_true, y_pred)
        metadata_all = metadata_dict(idx, responses_set)

if __name__ == '__main__':
    check_custom_integration()



