import tensorflow as tf
import numpy.typing as npt
import json
from typing import Dict, Any, Union
from PIL import Image

from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.visualizer_classes import LeapImage, LeapImageMask
from code_loader.contract.enums import (
    LeapDataType
)
from numpy import ndarray

from domain_gap.configs import *
from domain_gap.data.cs_data import CATEGORIES
from domain_gap.data.preprocess import subset_images
from domain_gap.visualizers.visualizers import get_loss_overlayed_img, get_cityscape_mask_img, get_masked_img
from domain_gap.visualizers.visualizers_utils import unnormalize_image
from domain_gap.metrics import mean_iou, class_mean_iou
from domain_gap.utils.gcs_utils import _download
from domain_gap.configs import IMAGE_SIZE, AUGMENT, TRAIN_SIZE
from domain_gap.data.cs_data import Cityscapes


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

def get_categorical_mask(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data
    cloud_path = data['gt_path'][idx % data["real_size"]]
    fpath = _download(cloud_path)
    mask = np.array(Image.open(fpath).resize(IMAGE_SIZE, Image.Resampling.NEAREST))
    if data['dataset'][idx % data["real_size"]] == 'cityscapes_od':
        encoded_mask = Cityscapes.encode_target_cityscapes(mask)
    else:
        encoded_mask = Cityscapes.encode_target(mask)
    return encoded_mask


def get_metadata_json(idx: int, data: PreprocessResponse) -> Dict[str, str]:
    cloud_path = data.data['metadata'][idx]
    fpath = _download(cloud_path)
    with open(fpath, 'r') as f:
        metadata_dict = json.loads(f.read())
    return metadata_dict


def metadata_idx(idx: int, data: PreprocessResponse) -> int:
    """ add TL index """
    return idx


def metadata_class_percent(idx: int, data: PreprocessResponse) -> dict:
    res = {}
    mask = get_categorical_mask(idx % data.data["real_size"], data)
    unique, counts = np.unique(mask, return_counts=True)
    unique_per_obj = dict(zip(unique, counts))
    for i, c in enumerate(CATEGORIES + ["background"]):
        count_obj = unique_per_obj.get(float(i))
        if count_obj is not None:
            percent_obj = count_obj / mask.size
        else:
            percent_obj = 0.0
        res[f'{c}'] = percent_obj
    return res


def metadata_brightness(idx: int, data: PreprocessResponse) -> ndarray:
    img = non_normalized_input_image(idx % data.data["real_size"], data)
    return np.mean(img)

def metadata_filename_city_dataset(idx: int, data: PreprocessResponse) -> dict[str, Any]:
    res = {'file_names': data.data['file_names'][idx],
           'cities': data.data['cities'][idx],
           'dataset': data.data['dataset'][idx % data.data["real_size"]]
           }
    return res


def metadata_json_data(idx: int, data: PreprocessResponse) -> dict[str, Union[str, Any]]:
    json_data = get_metadata_json(idx, data)
    gpsHeading = json_data['gpsHeading'] if data.data['dataset'][idx] == "cityscapes_od" else DEFAULT_GPS_HEADING
    gpsLatitude = json_data['gpsLatitude'] if data.data['dataset'][idx] == "cityscapes_od" else DEFAULT_GPS_LATITUDE
    gpsLongitude = json_data['gpsLongitude'] if data.data['dataset'][idx] == "cityscapes_od" else DEFAULT_GPS_LONGTITUDE
    outsideTemperature = json_data['outsideTemperature'] if data.data['dataset'][
                                                                idx] == "cityscapes_od" else DEFAULT_TEMP
    speed = json_data['speed'] if data.data['dataset'][idx] == "cityscapes_od" else DEFAULT_SPEED
    yawRate = json_data['yawRate'] if data.data['dataset'][idx] == "cityscapes_od" else DEFAULT_YAW_RATE

    res = {'gpsHeading': gpsHeading,
           'gpsLatitude': gpsLatitude,
           'gpsLongitude': gpsLongitude,
           'outsideTemperature': outsideTemperature,
           'speed': speed,
           'yawRate': yawRate
           }

    return res

# ----------------------------------- Visualizers ------------------------------------------


def image_visualizer(image: npt.NDArray[np.float32]) -> LeapImage:
    return LeapImage((unnormalize_image(image) * 255).astype(np.uint8))


def mask_visualizer(image: npt.NDArray[np.float32], mask: npt.NDArray[np.uint8]) -> LeapImageMask:
    mask = get_masked_img(image, mask)
    return LeapImageMask(mask.astype(np.uint8), unnormalize_image(image).astype(np.float32), CATEGORIES + ["excluded"])


def cityscape_segmentation_visualizer(mask: npt.NDArray[np.uint8]) -> LeapImage:
    mask_image = get_cityscape_mask_img(mask)
    return LeapImage(mask_image.astype(np.uint8))


def loss_visualizer(image: npt.NDArray[np.float32], prediction: npt.NDArray[np.float32],
                    gt: npt.NDArray[np.float32]) -> LeapImage:
    overlayed_image = get_loss_overlayed_img(image, prediction, gt)
    return LeapImage(overlayed_image)


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
leap_binder.set_metadata(metadata_idx, 'idx')
leap_binder.set_metadata(metadata_brightness, 'brightness')
leap_binder.set_metadata(metadata_filename_city_dataset, 'filename_city_dataset')
leap_binder.set_metadata(metadata_json_data, 'json_data')

leap_binder.set_visualizer(image_visualizer, 'image_visualizer', LeapDataType.Image)
leap_binder.set_visualizer(mask_visualizer, 'mask_visualizer', LeapDataType.ImageMask)
leap_binder.set_visualizer(cityscape_segmentation_visualizer, 'cityscapes_visualizer', LeapDataType.Image)
leap_binder.set_visualizer(loss_visualizer, 'loss_visualizer', LeapDataType.Image)

leap_binder.add_prediction('seg_mask', CATEGORIES)
