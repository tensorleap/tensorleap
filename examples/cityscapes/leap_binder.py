from typing import List, Dict, Callable, Union
from PIL import Image
import numpy as np
import json
import tensorflow as tf
from tensorflow import Tensor
from utils_all.gcs_utils import _download
from utils_all.metrics import regression_metric, classification_metric, object_metric, od_loss, calculate_iou
from utils_all.preprocessing import Cityscapes, load_cityscapes_data, CATEGORIES, CATEGORIES_no_background, \
    CATEGORIES_id_no_background
from config import CONFIG
from utils_all.general_utils import extract_bounding_boxes_from_instance_segmentation_polygons
from visualizers.visualizers import bb_decoder, gt_bb_decoder, bb_car_gt_decoder, bb_car_decoder

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
            "dataset": ["cityscapes"] * lengths[i]
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

def number_of_bb(index: int, subset: PreprocessResponse) -> int:
    bbs = np.array(ground_truth_bbox(index, subset))
    number_of_bb = np.count_nonzero(bbs[..., -1] != CONFIG['BACKGROUND_LABEL'])
    return number_of_bb

def instances_num(index: int, subset: PreprocessResponse) -> float:
    bbs = np.array(ground_truth_bbox(index, subset))
    valid_bbs = bbs[bbs[..., -1] != CONFIG['BACKGROUND_LABEL']]
    return float(valid_bbs.shape[0])

def avg_bb_aspect_ratio(index: int, subset: PreprocessResponse) -> float:
    bbs = np.array(ground_truth_bbox(index, subset))
    valid_bbs = bbs[bbs[..., -1] != CONFIG['BACKGROUND_LABEL']]
    assert ((valid_bbs[:, 3] > 0).all())
    aspect_ratios = valid_bbs[:, 2] / valid_bbs[:, 3]
    return aspect_ratios.mean()

def avg_bb_area_metadata(index: int, subset: PreprocessResponse) -> float:
    bbs = np.array(ground_truth_bbox(index, subset)) # x,y,w,h
    valid_bbs = bbs[bbs[..., -1] != CONFIG['BACKGROUND_LABEL']]
    areas = valid_bbs[:, 2] * valid_bbs[:, 3]
    return areas.mean()

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

def is_class_exist_veg_and_building(class_id_veg: int, class_id_building) -> Callable[[int, PreprocessResponse], float]:
    def func(index: int, subset: PreprocessResponse):
        bbs = np.array(ground_truth_bbox(index, subset))
        is_veg_exist = (bbs[..., -1] == class_id_veg).any()
        is_building_exist = (bbs[..., -1] == class_id_building).any()
        return is_veg_exist and is_building_exist

    func.__name__ = f'metadata_class_veg_and_class_building_instances_count'
    return func


def get_class_mean_iou(class_id: int) -> Callable[[Tensor, Tensor], Tensor]:
    def class_mean_iou(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        iou = calculate_iou(y_true, y_pred, class_id)
        return tf.convert_to_tensor(np.array([iou]), dtype=tf.float32)

    return class_mean_iou

def count_small_bbs(idx: int, data: PreprocessResponse) -> float:
    bboxes = np.array(ground_truth_bbox(idx, data))
    areas = bboxes[..., 2] * bboxes[..., 3]
    return float(len(areas[areas < CONFIG['SMALL_BBS_TH']]))

def metadata_filename(idx: int, data: PreprocessResponse) -> str:
    return data.data['file_names'][idx]

def metadata_city(idx: int, data: PreprocessResponse) -> str:
    return data.data['cities'][idx]

def metadata_idx(idx: int, data: PreprocessResponse) -> int:
    return idx

def get_metadata_json(idx: int, data: PreprocessResponse) -> Dict[str,str]:
    cloud_path = data.data['metadata'][idx]
    fpath = _download(cloud_path)
    with open(fpath, 'r') as f:
        metadata_dict = json.loads(f.read())
    return metadata_dict

def metadata_gps_heading(idx: int, data: PreprocessResponse) -> float:
    return get_metadata_json(idx, data)['gpsHeading']

def metadata_gps_latitude(idx: int, data: PreprocessResponse) -> float:
    return get_metadata_json(idx, data)['gpsLatitude']

def metadata_gps_longtitude(idx: int, data: PreprocessResponse) -> float:
    return get_metadata_json(idx, data)['gpsLongitude']

def metadata_outside_temperature(idx: int, data: PreprocessResponse) -> float:
    return get_metadata_json(idx, data)['outsideTemperature']

def metadata_speed(idx: int, data: PreprocessResponse) -> float:
    return get_metadata_json(idx, data)['speed']

def metadata_yaw_rate(idx: int, data: PreprocessResponse) -> float:
    return get_metadata_json(idx, data)['yawRate']

def metadata_brightness(idx: int, data: PreprocessResponse) -> float:
    img = non_normalized_image(idx, data)
    return np.mean(img)

def category_percent(idx: int, data: PreprocessResponse, class_id:int) -> float:
    bbs = np.array(ground_truth_bbox(idx, data))
    valid_bbs = bbs[bbs[..., -1] != CONFIG['BACKGROUND_LABEL']]
    category_bbs = valid_bbs[valid_bbs[..., -1] == class_id]
    return float(category_bbs.shape[0])

def metadata_person_category_avg_size(idx: int, data: PreprocessResponse) -> float:
    class_id = 24
    percent_val = category_percent(idx, data, class_id)
    instances_cnt = number_of_bb(idx, data)
    return np.round(percent_val/instances_cnt, 3) if instances_cnt > 0 else 0

def metadata_car_category_avg_size(idx: int, data: PreprocessResponse) -> float:
    class_id = 26
    car_val = category_percent(idx, data, class_id)
    instances_cnt = number_of_bb(idx, data)
    return np.round(car_val/instances_cnt, 3) if instances_cnt > 0 else 0

def metadata_dict(idx: int, data: PreprocessResponse) -> Dict[str, Union[float, int, str]]:
    metadata_functions = {
        "filename": metadata_filename,
        "city": metadata_city,
        "idx": metadata_idx,
        "gps_heading": metadata_gps_heading,
        "gps_latitude": metadata_gps_latitude,
        "gps_longtitude": metadata_gps_longtitude,
        "outside_temperature": metadata_outside_temperature,
        "speed": metadata_speed,
        "yaw_rate": metadata_yaw_rate,
        "bb_count": number_of_bb,
        "avg_bb_aspect_ratio": avg_bb_aspect_ratio,
        "avg_bb_area": avg_bb_area_metadata,
        "instances_number_metadata": instances_num,
        "small_bbs_number": count_small_bbs,
        "does_veg_and_buildeng_class_exist": is_class_exist_veg_and_building(21, 11),
        "metadata_brightness": metadata_brightness,
        "metadata_person_category_avg_size": metadata_person_category_avg_size,
        "metadata_car_category_avg_size": metadata_car_category_avg_size,

    }
    for i, label in enumerate(CATEGORIES_no_background):
        metadata_functions[f'{label} number_metadata'] = label_instances_num(label)

    for id in CATEGORIES_id_no_background:
        metadata_functions[f'does_class_number_{id}_exist'] = is_class_exist_gen(id)

    res = dict()
    for func_name, func in metadata_functions.items():
        res[func_name] = func(idx, data)
    return res

def od_metrics_dict(bb_gt: tf.Tensor, detection_pred: tf.Tensor) -> Dict[str, tf.Tensor]:
    metric_functions = {
        "Regression_metric": regression_metric,
        "Classification_metric": classification_metric,
        "Objectness_metric": object_metric,
    }

    for id in CATEGORIES_id_no_background:
        class_name = Cityscapes.get_class_name(id)
        metric_functions[f"iou_class_{class_name}"] = get_class_mean_iou(id)

    res = dict()
    for func_name, func in metric_functions.items():
        res[func_name] = func(bb_gt, detection_pred)
    return res

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
leap_binder.set_metadata(metadata_dict, name='metadata')

#set visualizer
leap_binder.set_visualizer(gt_bb_decoder, 'bb_gt_decoder', LeapDataType.ImageWithBBox)
leap_binder.set_visualizer(bb_decoder, 'bb_decoder', LeapDataType.ImageWithBBox)
leap_binder.set_visualizer(bb_car_gt_decoder, 'bb_car_gt_decoder', LeapDataType.ImageWithBBox)
leap_binder.set_visualizer(bb_car_decoder, 'bb_car_decoder', LeapDataType.ImageWithBBox)

# set custom metrics
leap_binder.add_custom_metric(od_metrics_dict, 'od_metrics')



