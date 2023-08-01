from typing import List, Dict, Callable
from PIL import Image
import numpy as np
import json

from utils_all.gcs_utils import _download
from utils_all.metrics import regression_metric, classification_metric, object_metric, od_loss
from utils_all.preprocessing import Cityscapes, load_cityscapes_data, CATEGORIES, CATEGORIES_no_background, \
    CATEGORIES_id_no_background
from project_config import IMAGE_STD, IMAGE_MEAN, IMAGE_SIZE, BACKGROUND_LABEL, SMALL_BBS_TH
from utils_all.general_utils import polygon_to_bbox, extract_bounding_boxes_from_instance_segmentation_polygons
from visualizers.visualizers import bb_decoder, gt_bb_decoder

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

    responses = [PreprocessResponse(length=len(all_images[0]), data={
                "image_path": all_images[0],
                "subset_name": "train",
                "gt_path": all_gt_labels[0],
                "gt_bbx_path": all_gt_labels_for_bbx[0],
                "gt_image_path": all_gt_images[0],
                "real_size": len(all_images[0]),
                "file_names": all_file_names[0],
                "cities": all_cities[0],
                "metadata": all_metadata[0],
                "dataset": ["cityscapes"]*len(all_images[0])}),
                PreprocessResponse(length=len(all_images[1]), data={
                "image_path": all_images[1],
                "subset_name": "val",
                "gt_path": all_gt_labels[1],
                "gt_bbx_path": all_gt_labels_for_bbx[1],
                "gt_image_path": all_gt_images[1],
                "real_size": len(all_images[1]),
                "file_names": all_file_names[1],
                "cities": all_cities[1],
                "metadata": all_metadata[1],
                "dataset": ["cityscapes"]*len(all_images[1])}),
                PreprocessResponse(length=len(all_images[2]), data={
                "image_path": all_images[2],
                "subset_name": "test",
                "gt_path": all_gt_labels[2],
                "gt_bbx_path": all_gt_labels_for_bbx[2],
                "gt_image_path": all_gt_images[2],
                "real_size": len(all_images[2]),
                "file_names": all_file_names[2],
                "cities": all_cities[2],
                "metadata": all_metadata[2],
                "dataset": ["cityscapes"] * len(all_images[2])})]
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

# def metadata_brightness(idx: int, data: PreprocessResponse) -> float:
#     img = non_normalized_image(idx%data.data["real_size"], data)
#     return np.mean(img)


# def aug_factor_or_zero(idx: int, data: PreprocessResponse, value: float) -> float:
#     if data.data["subset_name"] == "train" and AUGMENT and idx > TRAIN_SIZE-1:
#         return value.numpy()
#     else:
#         return 0.

# def metadata_hue_factor(idx: int, data: PreprocessResponse) -> float:
#     factor = stateless_random_uniform(shape=[], minval=-HUE_LIM, maxval=HUE_LIM, seed=(idx, idx+1))
#     return aug_factor_or_zero(idx, data, factor)

# def metadata_saturation_factor(idx: int, data: PreprocessResponse) -> float:
#     factor = stateless_random_uniform(shape=[], minval=1-SATUR_LIM, maxval=1+SATUR_LIM, seed=(idx, idx+1))
#     return aug_factor_or_zero(idx, data, factor)


# def metadata_contrast_factor(idx: int, data: PreprocessResponse) -> float:
#     factor = stateless_random_uniform(shape=[], minval=1-CONTR_LIM, maxval=1+CONTR_LIM, seed=(idx, idx+1))
#     return aug_factor_or_zero(idx, data, factor)


# def metadata_brightness_factor(idx: int, data: PreprocessResponse) -> float:
#     factor = stateless_random_uniform(shape=[], minval=-BRIGHT_LIM, maxval=BRIGHT_LIM, seed=(idx, idx+1))
#     return aug_factor_or_zero(idx, data, factor)


# def metadata_person_category_avg_size(idx: int, data: PreprocessResponse) -> float:
#     percent_val = metadata_person_category_percent(idx, data)
#     instances_cnt = metadata_person_instances_count(idx, data)
#     return np.round(percent_val/instances_cnt, 3) if instances_cnt > 0 else 0
#
#
# def metadata_car_vehicle_category_avg_size(idx: int, data: PreprocessResponse) -> float:
#     percent_val = metadata_car_vehicle_category_percent(idx, data)
#     if SUPERCATEGORY_GROUNDTRUTH:
#         instances_cnt = metadata_vehicle_instances_count(idx, data)
#     else:
#         instances_cnt = metadata_car_instances_count(idx, data)
#     return np.round(percent_val/instances_cnt, 3) if instances_cnt > 0 else 0

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
    leap_binder.set_metadata(is_class_exist_gen(id), DatasetMetadataType.float, f'does_class_number_{id}_exist')

#set visualizer
leap_binder.set_visualizer(gt_bb_decoder, 'bb_gt_decoder', LeapDataType.ImageWithBBox)
leap_binder.set_visualizer(bb_decoder, 'bb_decoder', LeapDataType.ImageWithBBox)

# set custom metrics
leap_binder.add_custom_metric(regression_metric, "Regression_metric")
leap_binder.add_custom_metric(classification_metric, "Classification_metric")
leap_binder.add_custom_metric(object_metric, "Objectness_metric")



