from typing import Dict, Callable, Union
from PIL import Image
import json
from tensorflow import Tensor

from typing import List
import numpy as np
import tensorflow as tf

from cityscapes_od.config import CONFIG
from cityscapes_od.data.preprocess import load_cityscapes_data, CATEGORIES, CATEGORIES_no_background, \
    CATEGORIES_id_no_background, Cityscapes
from cityscapes_od.metrics import calculate_iou, od_loss, metric
from cityscapes_od.utils.gcs_utils import _download
from cityscapes_od.utils.general_utils import extract_bounding_boxes_from_instance_segmentation_polygons, \
    bb_array_to_object, get_predict_bbox_list, instances_num, avg_bb_aspect_ratio, avg_bb_area_metadata, \
    count_small_bbs, number_of_bb

from code_loader.contract.responsedataclasses import BoundingBox
from code_loader.contract.visualizer_classes import LeapImageWithBBox
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.enums import (
    LeapDataType
)

# ----------------------------------------------------data processing--------------------------------------------------
def load_cityscapes_data_leap() -> List[PreprocessResponse]:
    all_images, all_gt_images, all_gt_labels, all_gt_labels_for_bbx, all_file_names, all_metadata, all_cities =\
        load_cityscapes_data()

    # train_len = len(all_images[0])
    # val_len = len(all_images[1])
    # test_len = len(all_images[2])

    train_len = 700
    val_len = 100
    test_len = 200

    lengths = [train_len, val_len, test_len]
    responses = [
        PreprocessResponse(length=lengths[i], data={
            "image_path": all_images[i],
            "subset_name": ["train", "val", "test"][i],
            "gt_path": all_gt_labels[i],
            "gt_bbx_path": all_gt_labels_for_bbx[i],
            "gt_image_path": all_gt_images[i],
            "real_size": lengths[i],
            "file_names": all_file_names[i],
            "cities": all_cities[i],
            "metadata": all_metadata[i],
            "dataset": ["cityscapes_od"] * lengths[i]
        }) for i in range(3)
    ]
    return responses

#------------------------------------------input and gt------------------------------------------

def non_normalized_image(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data
    cloud_path = data['image_path'][idx]
    fpath = _download(str(cloud_path))
    img = np.array(Image.open(fpath).convert('RGB').resize(CONFIG['IMAGE_SIZE']))/255.
    return img

def ground_truth_bbox(idx: int, data: PreprocessResponse) -> np.ndarray:
    """
    Description: This function takes an integer index idx and a PreprocessResponse object data as input and returns an
                 array of bounding boxes representing ground truth annotations.

    Input: idx (int): sample index.
    data (PreprocessResponse): An object of type PreprocessResponse containing data attributes.
    Output: bounding_boxes (np.ndarray): An array of bounding boxes extracted from the instance segmentation polygons in
            the JSON data. Each bounding box is represented as an array containing [x_center, y_center, width, height, label].
    """
    data = data.data
    cloud_path = data['gt_bbx_path'][idx]
    fpath = _download(cloud_path)
    with open(fpath, 'r') as file:
        json_data = json.load(file)
    bounding_boxes = extract_bounding_boxes_from_instance_segmentation_polygons(json_data)
    return bounding_boxes

# ----------------------------------------------------------metadata----------------------------------------------------
def metadata_filename(idx: int, data: PreprocessResponse) -> str:
    return data.data['file_names'][idx]

def metadata_city(idx: int, data: PreprocessResponse) -> str:
    return data.data['cities'][idx]

def metadata_idx(idx: int, data: PreprocessResponse) -> int:
    return idx

def metadata_brightness(idx: int, data: PreprocessResponse) -> float:
    img = non_normalized_image(idx, data)
    return np.mean(img)

def get_metadata_json(idx: int, data: PreprocessResponse) -> Dict[str,str]:
    cloud_path = data.data['metadata'][idx]
    fpath = _download(cloud_path)
    with open(fpath, 'r') as f:
        metadata_dict = json.loads(f.read())
    return metadata_dict

def metadata_json(idx: int, data: PreprocessResponse):
    json_dict = get_metadata_json(idx, data)
    res = {
        "gps_heading": json_dict['gpsHeading'],
        "gps_latitude": json_dict['gpsLatitude'],
        "gps_longtitude": json_dict['gpsLongitude'],
        "outside_temperature": json_dict['outsideTemperature'],
        "speed": json_dict['speed'],
        "yaw_rate": json_dict['yawRate']
    }
    return res
#
def category_percent(idx: int, data: PreprocessResponse, class_id:int) -> float:
    bbs = np.array(ground_truth_bbox(idx, data))
    valid_bbs = bbs[bbs[..., -1] != CONFIG['BACKGROUND_LABEL']]
    category_bbs = valid_bbs[valid_bbs[..., -1] == class_id]
    return bbs, float(category_bbs.shape[0])

def category_avg_size(idx: int, data: PreprocessResponse, class_id: int) -> float:
    bbs, car_val = category_percent(idx, data, class_id)
    instances_cnt = number_of_bb(bbs)
    return np.round(car_val/instances_cnt, 3) if instances_cnt > 0 else 0

def metadata_category_avg_size(idx: int, data: PreprocessResponse) -> Dict[str, float]:
    res = {
        "metadata_person_category_avg_size": category_avg_size(idx, data, 24),
        "metadata_car_category_avg_size": category_avg_size(idx, data, 26)
    }
    return res

#
def metadata_bbs(idx: int, data: PreprocessResponse) -> Dict[str, Union[float, int, str]]:
    bboxes = np.array(ground_truth_bbox(idx, data))
    valid_bbs = bboxes[bboxes[..., -1] != CONFIG['BACKGROUND_LABEL']]
    res = {
    "instances_number": instances_num(valid_bbs),
    "bb_aspect_ratio": avg_bb_aspect_ratio(valid_bbs),
    "avg_bb_area": avg_bb_area_metadata(valid_bbs),
    "small_bbs": count_small_bbs(bboxes),
    "bbox_number": number_of_bb(bboxes)

    }
    return res
#
def label_instances_num(class_label: str) -> Callable[[int, PreprocessResponse], float]:
    def func(index: int, subset: PreprocessResponse) -> float:
        bbs = np.array(ground_truth_bbox(index, subset))
        label = CATEGORIES.index(class_label)
        valid_bbs = bbs[bbs[..., -1] == label]
        return float(valid_bbs.shape[0])

    func.__name__ = f'metadata_{class_label}_instances_count'
    return func

def is_class_exist_gen(class_id: int) -> Callable[[int, PreprocessResponse], float]:
    def func(index: int, subset: PreprocessResponse):
        bbs = np.array(ground_truth_bbox(index, subset))
        is_i_exist = (bbs[..., -1] == class_id).any()
        return is_i_exist

    func.__name__ = f'metadata_{class_id}_instances_count'
    return func

def is_class_exist_veg_and_building(class_id_veg: int, class_id_building: int) -> Callable[[int, PreprocessResponse], float]:
    def func(index: int, subset: PreprocessResponse):
        bbs = np.array(ground_truth_bbox(index, subset))
        is_veg_exist = (bbs[..., -1] == class_id_veg).any()
        is_building_exist = (bbs[..., -1] == class_id_building).any()
        return is_veg_exist and is_building_exist

    func.__name__ = f'metadata_class_veg_and_class_building_instances_count'
    return func

# ---------------------------------------------------------metrics------------------------------------------------------

def get_class_mean_iou(class_id: int) -> Callable[[Tensor, Tensor], Tensor]:
    def class_mean_iou(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        iou = calculate_iou(y_true, y_pred, class_id)
        return tf.convert_to_tensor(np.array([iou]), dtype=tf.float32)

    return class_mean_iou

def od_metrics_dict(bb_gt: tf.Tensor, detection_pred: tf.Tensor) -> Dict[str, tf.Tensor]:
    losses = metric(bb_gt, detection_pred)
    metric_functions = {
        "Regression_metric": losses[0],
        "Classification_metric": losses[1],
        "Objectness_metric": losses[2],
    }
    return metric_functions

def gt_bb_decoder(image: np.ndarray, bb_gt: tf.Tensor) -> LeapImageWithBBox:
    """
    This function overlays ground truth bounding boxes (BBs) on the input image.

    Parameters:
    image (np.ndarray): The input image for which the ground truth bounding boxes need to be overlaid.
    bb_gt (np.ndarray): The ground truth bounding box array for the input image.

    Returns:
    An instance of LeapImageWithBBox containing the input image with ground truth bounding boxes overlaid.
    """
    bb_object: List[BoundingBox] = bb_array_to_object(bb_gt, iscornercoded=False, bg_label=CONFIG['BACKGROUND_LABEL'],
                                                      is_gt=True)
    bb_object = [bbox for bbox in bb_object if bbox.label in CATEGORIES_no_background]
    return LeapImageWithBBox(data=(image * 255).astype(np.float32), bounding_boxes=bb_object)


def bb_car_gt_decoder(image: np.ndarray, bb_gt: tf.Tensor) -> LeapImageWithBBox:
    """
    Overlays the BB predictions on the image
    """
    bb_object: List[BoundingBox] = bb_array_to_object(bb_gt, iscornercoded=False, bg_label=CONFIG['BACKGROUND_LABEL'], is_gt=True)
    bb_object = [bbox for bbox in bb_object if bbox.label == 'car']
    return LeapImageWithBBox(data=(image * 255).astype(np.float32), bounding_boxes=bb_object)

def bb_decoder(image: np.ndarray, predictions: tf.Tensor) -> LeapImageWithBBox:
    """
    Overlays the BB predictions on the image
    """
    bb_object = get_predict_bbox_list(predictions)
    bb_object = [bbox for bbox in bb_object if bbox.label in CATEGORIES_no_background]
    return LeapImageWithBBox(data=(image * 255).astype(np.float32), bounding_boxes=bb_object)

def bb_car_decoder(image: np.ndarray, predictions: tf.Tensor) -> LeapImageWithBBox:
    """
    Overlays the BB predictions on the image
    """
    bb_object = get_predict_bbox_list(predictions)
    bb_object = [bbox for bbox in bb_object if bbox.label == 'car']
    return LeapImageWithBBox(data=(image * 255).astype(np.float32), bounding_boxes=bb_object)

# ---------------------------------------------------------binding------------------------------------------------------

#preprocess function
leap_binder.set_preprocess(load_cityscapes_data_leap)

#set input and gt
leap_binder.set_input(non_normalized_image, 'non_normalized_image')
leap_binder.set_ground_truth(ground_truth_bbox, 'bbox')

#set prediction
leap_binder.add_prediction(name='object detection', labels=["x", "y", "w", "h", "obj"] + [cl for cl in CATEGORIES])

#set loss
leap_binder.add_custom_loss(od_loss, 'od_loss')

#set meata_data
leap_binder.set_metadata(metadata_filename, name='metadata_filename')
leap_binder.set_metadata(metadata_city, name='metadata_city')
leap_binder.set_metadata(metadata_idx, name='metadata_idx')
leap_binder.set_metadata(metadata_brightness, name='metadata_brightness')
leap_binder.set_metadata(metadata_json, name='metadata_json')
leap_binder.set_metadata(metadata_category_avg_size, name='metadata_category_avg_size')
leap_binder.set_metadata(metadata_bbs, name='metadata_bbs')
for label in CATEGORIES_no_background:
    leap_binder.set_metadata(label_instances_num(label), f'{label} number_metadata')
for id in CATEGORIES_id_no_background:
    class_name = Cityscapes.get_class_name(id)
    leap_binder.set_metadata(is_class_exist_gen(id), f'does_class_number_{id}_exist')
leap_binder.set_metadata(is_class_exist_veg_and_building(21, 11), "does_veg_and_buildeng_class_exist")


#set visualizer
leap_binder.set_visualizer(gt_bb_decoder, 'bb_gt_decoder', LeapDataType.ImageWithBBox)
leap_binder.set_visualizer(bb_decoder, 'bb_decoder', LeapDataType.ImageWithBBox)
leap_binder.set_visualizer(bb_car_gt_decoder, 'bb_car_gt_decoder', LeapDataType.ImageWithBBox)
leap_binder.set_visualizer(bb_car_decoder, 'bb_car_decoder', LeapDataType.ImageWithBBox)

# set custom metrics
leap_binder.add_custom_metric(od_metrics_dict, 'od_metrics')
for id in CATEGORIES_id_no_background:
    class_name = Cityscapes.get_class_name(id)
    leap_binder.add_custom_metric(get_class_mean_iou(id), f"iou_class_{class_name}")



