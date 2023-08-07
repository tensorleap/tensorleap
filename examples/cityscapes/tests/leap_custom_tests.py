import json
import os
from typing import List, Tuple

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import webcolors

from project_config import MAX_BB_PER_IMAGE
from utils_all.gcs_utils import _download
from utils_all.general_utils import filter_out_unknown_classes_id, normelized_polygon
from utils_all.metrics import od_loss, object_metric, classification_metric, regression_metric
from utils_all.preprocessing import Cityscapes, CATEGORIES_no_background, CATEGORIES_id_no_background
from visualizers.visualizers import gt_bb_decoder, bb_decoder, bb_car_decoder, bb_car_gt_decoder

from leap_binder import load_cityscapes_data_leap, metadata_filename, metadata_city, metadata_idx, metadata_gps_heading, \
    metadata_gps_latitude, metadata_gps_longtitude, metadata_outside_temperature, metadata_speed, metadata_yaw_rate, \
    number_of_bb, avg_bb_area_metadata, instances_num, is_class_exist_gen, ground_truth_bbox, \
    avg_bb_aspect_ratio, label_instances_num, non_normalized_image, count_small_bbs, is_class_exist_veg_and_building, \
    metadata_brightness, metadata_person_category_avg_size, metadata_car_category_avg_size, get_class_mean_iou
from code_loader.contract.datasetclasses import PreprocessResponse

def rgb_to_color_name(rgb_value: Tuple[int]) ->str:
    """
    Description: This function takes an RGB color value as input and converts it into a color name.

    Input: rgb_value (Tuple[int]): An RGB color value represented as a tuple of three integers (red, green, blue).
    Output: color_name (str): The color name corresponding to the input RGB value or 'r' as a default placeholder.
    """
    try:
        color_name = webcolors.rgb_to_name(rgb_value)
    except ValueError:
        color_name = 'r'
    return color_name

def plot_image_with_bboxes(image: np.ndarray, bounding_boxes: np.ndarray, type: str):
    """
    Description: The function takes an image and a list of bounding boxes as input and visualizes the image with
    bounding boxes overlaid.

    Input: image (numpy array): Input RGB image as a NumPy array.
           bounding_boxes (list): List of bounding boxes represented as [x_center, y_center, width, height, label].
           type (str): The bboxes type- gt ot prediction.
    Output: None. The function directly displays the image with overlaid bounding boxes using Matplotlib.
    """
    # Create a figure and axis
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image)

    # Add bounding boxes to the plot
    for bbox in bounding_boxes:
        label = bbox.label
        class_id = Cityscapes.get_class_id(label)
        color = Cityscapes.get_class_color(class_id)
        color_name = rgb_to_color_name(color)

        if label != 'unlabeled':
            x_center, y_center, width, height = bbox.x, bbox.y, bbox.width, bbox.height
            x_min = x_center - width / 2
            y_max = y_center + height / 2
            # Convert relative coordinates to absolute coordinates
            x_abs = x_min * image.shape[1]
            y_abs = y_max * image.shape[0]
            width_abs = width * image.shape[1]
            height_abs = -(height * image.shape[0])

            # Create a rectangle patch and add it to the plot
            rect = patches.Rectangle((x_abs, y_abs), width_abs, height_abs, linewidth=1, edgecolor=color_name, facecolor='none')
            ax.add_patch(rect)

            # Add label text to the rectangle
            plt.text(x_abs, y_abs, label, color=color_name, fontsize=8, backgroundcolor='white')

    # Show the plot
    plt.title(f"Image with {type} bboxes")
    plt.show()

def plot_image_with_polygons(image_height: int, image_width: int, polygons, image: np.ndarray):
    """
    Description: The function takes an image and a list of polygons as input and visualizes the image with
    polygons overlaid.

    Input: image (numpy array): Input RGB image as a NumPy array.
           image_height :(int): Height of the input image in pixels.
           image_width: (int): Width of the input image in pixels.
           polygons (list): List of polygons represented as dictionaries, with 'label' (int) and 'polygon' (list) keys.
    Output: None. The function directly displays the image with overlaid polygons using Matplotlib.
    """
    # Create a figure and axis
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image)

    # Add polygons to the plot
    for polygon in polygons:
        polygon = normelized_polygon(image_height, image_width, polygon)
        label = polygon['label']
        color = Cityscapes.get_class_color(label)
        color_name = rgb_to_color_name(color)

        class_name = Cityscapes.get_class_name(label)
        if class_name == 'car':
            coords = polygon['polygon']

            # Create a polygon patch and add it to the plot
            poly_patch = patches.Polygon(coords, linewidth=1, edgecolor=color_name, facecolor='none')
            ax.add_patch(poly_patch)

            # Add label text to the polygon
            centroid = [sum(coord[0] for coord in coords) / len(coords), sum(coord[1] for coord in coords) / len(coords)]
            plt.text(centroid[0], centroid[1], class_name, color=color_name, fontsize=8, backgroundcolor='white')

    plt.title("Image with Polygons")
    plt.show()

def get_json(idx: int, data: PreprocessResponse) -> dict:
    """
    Description: This function takes an integer index idx and a PreprocessResponse object data as input and returns a
                Python dictionary containing JSON data.

    Input: idx (int): Index of the sample.
    data (PreprocessResponse): An object of type PreprocessResponse containing data attributes.
    Output: json_data (dict): A Python dictionary representing the JSON data obtained from the file at the given index.
    """
    data = data.data
    cloud_path = data['gt_bbx_path'][idx%data["real_size"]]
    fpath = _download(cloud_path)
    with open(fpath, 'r') as file:
        json_data = json.load(file)
    return json_data

def get_polygon(json_data: dict) -> List[dict]:
    """
    Description: This function takes a Python dictionary json_data as input and returns a list of dictionaries
    representing polygons.

    Input: json_data (dict): A Python dictionary representing the JSON data containing annotation information.
    Output: polygons (List[dict]): A list of dictionaries, each representing a label polygon.
    """
    polygons = []
    objects = json_data['objects']
    objects = filter_out_unknown_classes_id(objects)
    max_anns = min(MAX_BB_PER_IMAGE, len(objects))
    for i in range(max_anns):
        polygon = objects[i]
        polygons.append(polygon)
    return polygons

def check_custom_integration():
    # preprocess function
    responses = load_cityscapes_data_leap()
    train = responses[0]
    val = responses[1]
    test = responses[2]
    responses_set = test
    for idx in range(20, 40):
        # get input and gt
        image = non_normalized_image(idx, responses_set)

        json_data = get_json(idx, responses_set)
        image_height, image_width = json_data['imgHeight'], json_data['imgWidth']
        polygons = get_polygon(json_data) #till here all equal
        plot_image_with_polygons(image_height, image_width, polygons, image)

        bounding_boxes_gt = ground_truth_bbox(idx, responses_set)

        # model
        path = "/Users/chenrothschild/repo/tensorleap/examples/cityscapes/model"
        os.chdir(path)
        model = os.path.join(path, 'exported-model-(8).h5')
        yolo = tf.keras.models.load_model(model)

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
        Regression_metric = regression_metric(y_true, y_pred)
        Classification_metric = classification_metric(y_true, y_pred)
        Object_metric = object_metric(y_true, y_pred)
        print(f"regression loss is {tf.stack(Regression_metric).numpy()}")
        print(f"classification loss is {tf.stack(Classification_metric).numpy()}")
        print(f"object loss is {tf.stack(Object_metric).numpy()}")

        # get meata_data
        #iou = calculate_iou(y_true, y_pred)
        for class_id in CATEGORIES_id_no_background:
            iou_id = get_class_mean_iou(class_id)
            iou_id_ = iou_id(y_true, y_pred)
            print(f"iou_id of {class_id} is {tf.stack(iou_id_).numpy()}")


        brightness = metadata_brightness(idx, responses_set)
        person_category_avg = metadata_person_category_avg_size(idx, responses_set)
        print(f'person_category_avg: {person_category_avg}')
        car_category_avg = metadata_car_category_avg_size(idx, responses_set)
        print(f'car_category_avg: {car_category_avg}')
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
        small_bbs_number = count_small_bbs(idx, responses_set)
        for class_label in CATEGORIES_no_background:
            instances_count_func = label_instances_num(class_label)
            instances_count = instances_count_func(idx, responses_set)
        for class_id in CATEGORIES_id_no_background:
            class_exist_func = is_class_exist_gen(class_id)
            class_exist = class_exist_func(idx, responses_set)
        vb_class_exist_func = is_class_exist_veg_and_building(21, 11)
        class_exist = vb_class_exist_func(idx, responses_set)

if __name__ == '__main__':
    check_custom_integration()



