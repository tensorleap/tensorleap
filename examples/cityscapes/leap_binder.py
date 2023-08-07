from typing import List, Dict, Callable
from PIL import Image
import numpy as np
import json
import tensorflow as tf

from utils_all.gcs_utils import _download
from utils_all.metrics import regression_metric, classification_metric, object_metric, od_loss, calculate_iou, \
    convert_to_xyxy
from utils_all.preprocessing import Cityscapes, load_cityscapes_data, CATEGORIES, CATEGORIES_no_background, \
    CATEGORIES_id_no_background
from project_config import IMAGE_STD, IMAGE_MEAN, IMAGE_SIZE, BACKGROUND_LABEL, SMALL_BBS_TH
from utils_all.general_utils import extract_bounding_boxes_from_instance_segmentation_polygons, \
    bb_array_to_object, get_predict_bbox_list
from visualizers.visualizers import bb_decoder, gt_bb_decoder, bb_car_gt_decoder, bb_car_decoder

from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.enums import (
    DatasetMetadataType,
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

    responses = [PreprocessResponse(length=train_len, data={
                "image_path": all_images[0],
                "subset_name": "train",
                "gt_path": all_gt_labels[0],
                "gt_bbx_path": all_gt_labels_for_bbx[0],
                "gt_image_path": all_gt_images[0],
                "real_size": train_len,
                "file_names": all_file_names[0],
                "cities": all_cities[0],
                "metadata": all_metadata[0],
                "dataset": ["cityscapes"]*train_len}),
                PreprocessResponse(length=val_len, data={
                "image_path": all_images[1],
                "subset_name": "val",
                "gt_path": all_gt_labels[1],
                "gt_bbx_path": all_gt_labels_for_bbx[1],
                "gt_image_path": all_gt_images[1],
                "real_size": val_len,
                "file_names": all_file_names[1],
                "cities": all_cities[1],
                "metadata": all_metadata[1],
                "dataset": ["cityscapes"]*val_len}),
                PreprocessResponse(length=test_len, data={
                "image_path": all_images[2],
                "subset_name": "test",
                "gt_path": all_gt_labels[2],
                "gt_bbx_path": all_gt_labels_for_bbx[2],
                "gt_image_path": all_gt_images[2],
                "real_size": test_len,
                "file_names": all_file_names[2],
                "cities": all_cities[2],
                "metadata": all_metadata[2],
                "dataset": ["cityscapes"] * test_len})]
    return responses

#------------------------------------------input and gt------------------------------------------

def non_normalized_image(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data
    cloud_path = data['image_path'][idx%data["real_size"]]
    fpath = _download(str(cloud_path))
    img = np.array(Image.open(fpath).convert('RGB').resize(IMAGE_SIZE))/255.
    return img


def input_image(idx: int, data: PreprocessResponse) -> np.ndarray:
    img = non_normalized_image(idx%data.data["real_size"], data)
    normalized_image = (img - IMAGE_MEAN)/IMAGE_STD #TODO: needed??
    #resized_image = zoom(normalized_image, (3.2, 1.6, 1))
    return normalized_image.astype(float)


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
    cloud_path = data['gt_bbx_path'][idx%data["real_size"]]
    fpath = _download(cloud_path)
    with open(fpath, 'r') as file:
        json_data = json.load(file)
    bounding_boxes = extract_bounding_boxes_from_instance_segmentation_polygons(json_data)
    return bounding_boxes

# ----------------------------------------------------------metadata----------------------------------------------------

def number_of_bb(index: int, subset: PreprocessResponse) -> int:
    bbs = np.array(ground_truth_bbox(index, subset))
    number_of_bb = np.count_nonzero(bbs[..., -1] != BACKGROUND_LABEL)
    return number_of_bb

def instances_num(index: int, subset: PreprocessResponse) -> float:
    bbs = np.array(ground_truth_bbox(index, subset))
    valid_bbs = bbs[bbs[..., -1] != BACKGROUND_LABEL]
    return float(valid_bbs.shape[0])

def avg_bb_aspect_ratio(index: int, subset: PreprocessResponse) -> float:
    bbs = np.array(ground_truth_bbox(index, subset))
    valid_bbs = bbs[bbs[..., -1] != BACKGROUND_LABEL]
    assert ((valid_bbs[:, 3] > 0).all())
    aspect_ratios = valid_bbs[:, 2] / valid_bbs[:, 3]
    return aspect_ratios.mean()

def avg_bb_area_metadata(index: int, subset: PreprocessResponse) -> float:
    bbs = np.array(ground_truth_bbox(index, subset)) # x,y,w,h
    valid_bbs = bbs[bbs[..., -1] != BACKGROUND_LABEL]
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


def get_class_mean_iou(class_id: int = None) -> tf.Tensor:

    def class_mean_iou(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calculate the mean Intersection over Union (mIOU) for segmentation using TensorFlow.

        Args:
            y_true (tf.Tensor): Ground truth segmentation mask tensor.
            y_pred (tf.Tensor): Predicted segmentation mask tensor.

        Returns:
            tf.Tensor: Mean Intersection over Union (mIOU) value.
        """
        y_true = bb_array_to_object(y_true, iscornercoded=False, bg_label=BACKGROUND_LABEL, is_gt=True)
        y_true = [bbox for bbox in y_true if bbox.label in CATEGORIES_no_background]
        y_true = convert_to_xyxy(y_true)

        y_pred = y_pred[0, ...]
        y_pred = get_predict_bbox_list(y_pred)
        y_pred = [bbox for bbox in y_pred if bbox.label in CATEGORIES_no_background]
        y_pred = convert_to_xyxy(y_pred)

        y_true = [box for box in y_true if box[-1] == class_id]
        y_pred = [box for box in y_pred if box[-1] == class_id]
        iou = calculate_iou(y_true, y_pred)

        return iou

    return class_mean_iou

def count_small_bbs(idx: int, data: PreprocessResponse) -> float:
    bboxes = np.array(ground_truth_bbox(idx, data))
    #obj_boxes = bboxes[bboxes[..., -1] == 0]
    areas = bboxes[..., 2] * bboxes[..., 3]
    return float(len(areas[areas < SMALL_BBS_TH]))

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
    img = non_normalized_image(idx%data.data["real_size"], data)
    return np.mean(img)

def category_percent(idx: int, data: PreprocessResponse, class_id:int) -> float:
    bbs = np.array(ground_truth_bbox(idx, data))
    valid_bbs = bbs[bbs[..., -1] != BACKGROUND_LABEL]
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

# ---------------------------------------------------------binding------------------------------------------------------

#preprocess function
leap_binder.set_preprocess(load_cityscapes_data_leap)
#TODO: unlables data

#set input and gt
leap_binder.set_input(non_normalized_image, 'non_normalized_image')
leap_binder.set_ground_truth(ground_truth_bbox, 'bbox')

#set prediction
leap_binder.add_prediction(name='object detection', labels=["x", "y", "w", "h", "obj"] + [cl for cl in CATEGORIES])

#set loss
leap_binder.add_custom_loss(od_loss, 'od_loss')

#set meata_data
leap_binder.set_metadata(metadata_filename, DatasetMetadataType.string, 'filename')
leap_binder.set_metadata(metadata_city, DatasetMetadataType.string, 'city')
leap_binder.set_metadata(metadata_idx, DatasetMetadataType.float, 'idx')
leap_binder.set_metadata(metadata_gps_heading, DatasetMetadataType.float, 'gps_heading')
leap_binder.set_metadata(metadata_gps_latitude, DatasetMetadataType.float, 'gps_latitude')
leap_binder.set_metadata(metadata_gps_longtitude, DatasetMetadataType.float, 'gps_longtitude')
leap_binder.set_metadata(metadata_outside_temperature, DatasetMetadataType.float, 'outside_temperature')
leap_binder.set_metadata(metadata_speed, DatasetMetadataType.float, 'speed')
leap_binder.set_metadata(metadata_yaw_rate, DatasetMetadataType.float, 'yaw_rate')
leap_binder.set_metadata(number_of_bb, DatasetMetadataType.int, 'bb_count')
leap_binder.set_metadata(avg_bb_aspect_ratio, DatasetMetadataType.float, 'avg_bb_aspect_ratio')
leap_binder.set_metadata(avg_bb_area_metadata, DatasetMetadataType.float, 'avg_bb_area')
leap_binder.set_metadata(instances_num, DatasetMetadataType.float, "instances_number_metadata")
leap_binder.set_metadata(count_small_bbs, DatasetMetadataType.int, "small_bbs_number")
for i, label in enumerate(CATEGORIES_no_background):
    leap_binder.set_metadata(label_instances_num(label), DatasetMetadataType.float, f'{label} number_metadata')
for id in CATEGORIES_id_no_background:
    class_name = Cityscapes.get_class_name(id)
    leap_binder.set_metadata(is_class_exist_gen(id), DatasetMetadataType.float, f'does_class_number_{id}_exist')
    leap_binder.add_custom_metric(get_class_mean_iou(id), name=f"iou_class_{class_name}") #metric
leap_binder.set_metadata(is_class_exist_veg_and_building(21, 11), DatasetMetadataType.float, f'does_veg_and_buildeng_class_exist')
leap_binder.set_metadata(metadata_brightness, DatasetMetadataType.float, "metadata_brightness")
leap_binder.set_metadata(metadata_person_category_avg_size, DatasetMetadataType.float, "metadata_person_category_avg_size")
leap_binder.set_metadata(metadata_car_category_avg_size, DatasetMetadataType.float, "metadata_car_category_avg_size")


#set visualizer
leap_binder.set_visualizer(gt_bb_decoder, 'bb_gt_decoder', LeapDataType.ImageWithBBox)
leap_binder.set_visualizer(bb_decoder, 'bb_decoder', LeapDataType.ImageWithBBox)
leap_binder.set_visualizer(bb_car_gt_decoder, 'bb_car_gt_decoder', LeapDataType.ImageWithBBox)
leap_binder.set_visualizer(bb_car_decoder, 'bb_car_decoder', LeapDataType.ImageWithBBox)

# set custom metrics
leap_binder.add_custom_metric(regression_metric, "Regression_metric")
leap_binder.add_custom_metric(classification_metric, "Classification_metric")
leap_binder.add_custom_metric(object_metric, "Objectness_metric")


