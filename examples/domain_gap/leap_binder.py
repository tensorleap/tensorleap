from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse
import tensorflow as tf
from PIL import Image
from code_loader.contract.enums import (
    LeapDataType
)

from domain_gap.data.cs_data import CATEGORIES
from domain_gap.utils.configs import *
from domain_gap.utils.gcs_utils import _download
from examples.domain_gap.domain_gap.data.preprocess import subset_images
from domain_gap.tl_helpers.visualizers.visualizers import image_visualizer, loss_visualizer, mask_visualizer, \
    cityscape_segmentation_visualizer
from domain_gap.tl_helpers.utils import get_categorical_mask, get_metadata_json, class_mean_iou, mean_iou


# ----------------------------------- Input ------------------------------------------

def non_normalized_input_image(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data
    cloud_path = data['image_path'][idx % data["real_size"]]
    fpath = _download(str(cloud_path))
    img = np.array(Image.open(fpath).convert('RGB').resize(IMAGE_SIZE)) / 255.
    return img


def input_image(idx: int, data: PreprocessResponse) -> np.ndarray:
    img = non_normalized_input_image(idx % data.data["real_size"], data)
    if data.data['dataset'][idx % data.data["real_size"]] == 'kitti':
        img = (img - KITTI_MEAN) * CITYSCAPES_STD / KITTI_STD + CITYSCAPES_MEAN
    normalized_image = (img - IMAGE_MEAN) / IMAGE_STD
    return normalized_image.astype(float)


# ----------------------------------- GT ------------------------------------------

def ground_truth_mask(idx: int, data: PreprocessResponse) -> np.ndarray:
    mask = get_categorical_mask(idx % data.data["real_size"], data)
    return tf.keras.utils.to_categorical(mask, num_classes=20).astype(float)[...,
           :19]  # Remove background class from cross-entropy


# ----------------------------------- Metadata ------------------------------------------

def metadata_idx(idx: int, data: PreprocessResponse) -> int:
    """ add TL index """
    return idx


def metadata_class_percent(idx: int, data: PreprocessResponse) -> dict:
    res = {}
    mask = get_categorical_mask(idx % data.data["real_size"], data)
    unique, counts = np.unique(mask, return_counts=True)
    unique_per_obj = dict(zip(unique, counts))
    for i, c in enumerate(CATEGORIES+["background"]):
        count_obj = unique_per_obj.get(float(i))
        if count_obj is not None:
            percent_obj = count_obj / mask.size
        else:
            percent_obj = 0.0
        res[f'{c}'] = percent_obj
    return res


def metadata_brightness(idx: int, data: PreprocessResponse) -> float:
    img = non_normalized_input_image(idx % data.data["real_size"], data)
    return np.mean(img)


def metadata_filename(idx: int, data: PreprocessResponse) -> str:
    return data.data['file_names'][idx]


def metadata_city(idx: int, data: PreprocessResponse) -> str:
    return data.data['cities'][idx]


def metadata_dataset(idx: int, data: PreprocessResponse) -> str:
    return data.data['dataset'][idx % data.data["real_size"]]


def metadata_gps_heading(idx: int, data: PreprocessResponse) -> float:
    if data.data['dataset'][idx] == "cityscapes_od":
        return get_metadata_json(idx, data)['gpsHeading']
    else:
        return DEFAULT_GPS_HEADING


def metadata_gps_latitude(idx: int, data: PreprocessResponse) -> float:
    if data.data['dataset'][idx] == "cityscapes_od":
        return get_metadata_json(idx, data)['gpsLatitude']
    else:
        return DEFAULT_GPS_LATITUDE


def metadata_gps_longtitude(idx: int, data: PreprocessResponse) -> float:
    if data.data['dataset'][idx] == "cityscapes_od":
        return get_metadata_json(idx, data)['gpsLongitude']
    else:
        return DEFAULT_GPS_LONGTITUDE


def metadata_outside_temperature(idx: int, data: PreprocessResponse) -> float:
    if data.data['dataset'][idx] == "cityscapes_od":
        return get_metadata_json(idx, data)['outsideTemperature']
    else:
        return DEFAULT_TEMP


def metadata_speed(idx: int, data: PreprocessResponse) -> float:
    if data.data['dataset'][idx] == "cityscapes_od":
        return get_metadata_json(idx, data)['speed']
    else:
        return DEFAULT_SPEED


def metadata_yaw_rate(idx: int, data: PreprocessResponse) -> float:
    if data.data['dataset'][idx] == "cityscapes_od":
        return get_metadata_json(idx, data)['yawRate']
    else:
        return DEFAULT_YAW_RATE


# ----------------------------------- Binding ------------------------------------------

leap_binder.set_preprocess(subset_images)

if NORM_CS:
    leap_binder.set_input(input_image, 'normalized_image')
else:
    leap_binder.set_input(non_normalized_input_image, 'non_normalized')

leap_binder.set_ground_truth(ground_truth_mask, 'mask')

leap_binder.add_custom_metric(class_mean_iou, name=f"iou_class")
leap_binder.add_custom_metric(mean_iou, name=f"iou")

leap_binder.set_metadata(metadata_class_percent, 'class_percent')
leap_binder.set_metadata(metadata_filename, 'filename')
leap_binder.set_metadata(metadata_city, 'city')
leap_binder.set_metadata(metadata_dataset, 'dataset')
leap_binder.set_metadata(metadata_idx, 'idx')
leap_binder.set_metadata(metadata_gps_heading, 'gps_heading')
leap_binder.set_metadata(metadata_gps_latitude, 'gps_latitude')
leap_binder.set_metadata(metadata_gps_longtitude, 'gps_longtitude')
leap_binder.set_metadata(metadata_outside_temperature, 'outside_temperature')
leap_binder.set_metadata(metadata_speed, 'speed')
leap_binder.set_metadata(metadata_yaw_rate, 'yaw_rate')

leap_binder.set_visualizer(image_visualizer, 'image_visualizer', LeapDataType.Image)
leap_binder.set_visualizer(mask_visualizer, 'mask_visualizer', LeapDataType.ImageMask)
leap_binder.set_visualizer(cityscape_segmentation_visualizer, 'cityscapes_visualizer', LeapDataType.Image)
leap_binder.set_visualizer(loss_visualizer, 'loss_visualizer', LeapDataType.Image)

leap_binder.add_prediction('seg_mask', CATEGORIES)
