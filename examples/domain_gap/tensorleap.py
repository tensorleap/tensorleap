from typing import List, Optional, Dict
from code_loader import leap_binder
from code_loader.contract.enums import DatasetMetadataType, Metric
from code_loader.contract.datasetclasses import PreprocessResponse
import tensorflow as tf
from tensorflow.python.ops.stateless_random_ops import stateless_random_uniform
import os
from PIL import Image
import numpy as np
import numpy.typing as npt
import json

from matplotlib import colors
from matplotlib import cm as cmx
import matplotlib.pyplot as plt
from code_loader.contract.visualizer_classes import LeapImage, LeapImageMask
from code_loader.contract.enums import (
    LeapDataType
)

from domain_gap.utils.configs import *
from domain_gap.utils.gcs_utils import _connect_to_gcs_and_return_bucket, _download
from domain_gap.data.cs_data import get_cityscapes_data
from domain_gap.data.kitti_data import get_kitti_data




def subset_images() -> List[PreprocessResponse]:
    responses = get_cityscapes_data()
    data_dict = get_kitti_data()
    for i, title in enumerate(["train", "validation"]):
        responses[i].data['image_path'] += data_dict[title]['image_path']
        responses[i].data['gt_path'] += data_dict[title]['gt_path']
        responses[i].data['file_names'] += data_dict[title]['image_path']
        responses[i].data['cities'] += ["Karlsruhe"] * len(data_dict[title]['image_path'])
        responses[i].data['dataset'] += ['kitti'] * len(data_dict[title]['image_path'])
        responses[i].data['real_size'] += len(data_dict[title]['image_path'])
        responses[i].data['metadata'] += [""] * len(data_dict[title]['image_path'])
        responses[i].length += len(data_dict[title]['image_path'])
        responses[i].length = subset_sizes[i]
    return responses


def non_normalized_image(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data
    cloud_path = data['image_path'][idx % data["real_size"]]
    fpath = _download(str(cloud_path))
    img = np.array(Image.open(fpath).convert('RGB').resize(image_size)) / 255.
    return img


def input_image(idx: int, data: PreprocessResponse) -> np.ndarray:
    img = non_normalized_image(idx % data.data["real_size"], data)
    if data.data['dataset'][idx % data.data["real_size"]] == 'kitti':
        img = (img - KITTI_MEAN) * CITYSCAPES_STD / KITTI_STD + CITYSCAPES_MEAN
    normalized_image = (img - IMAGE_MEAN) / IMAGE_STD
    return normalized_image.astype(float)


def get_categorical_mask(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data
    cloud_path = data['gt_path'][idx % data["real_size"]]
    fpath = _download(cloud_path)
    mask = np.array(Image.open(fpath).resize(image_size, Image.Resampling.NEAREST))
    if data['dataset'][idx % data["real_size"]] == 'cityscapes':
        encoded_mask = Cityscapes.encode_target_cityscapes(mask)
    else:
        encoded_mask = Cityscapes.encode_target(mask)
    return encoded_mask


def ground_truth_mask(idx: int, data: PreprocessResponse) -> np.ndarray:
    mask = get_categorical_mask(idx % data.data["real_size"], data)
    return tf.keras.utils.to_categorical(mask, num_classes=20).astype(float)[..., :19]  # Remove background class from cross-entropy


def metadata_background_percent(idx: int, data: PreprocessResponse) -> float:
    mask = get_categorical_mask(idx % data.data["real_size"], data)
    unique, counts = np.unique(mask, return_counts=True)
    unique_per_obj = dict(zip(unique, counts))
    count_obj = unique_per_obj.get(19.)
    if count_obj is not None:
        percent_obj = count_obj / mask.size
    else:
        percent_obj = 0.0
    return percent_obj


def metadata_percent_function_generator(class_idx: int):
    def get_metadata_percent(idx: int, data: PreprocessResponse) -> float:
        mask = get_categorical_mask(idx % data.data["real_size"], data)
        unique, counts = np.unique(mask, return_counts=True)
        unique_per_obj = dict(zip(unique, counts))
        count_obj = unique_per_obj.get(float(class_idx))
        if count_obj is not None:
            percent_obj = count_obj / mask.size
        else:
            percent_obj = 0.0
        return percent_obj

    get_metadata_percent.__name__ = Cityscapes.train_id_to_label[class_idx] + "_" + "class_percent"
    return get_metadata_percent


def metadata_brightness(idx: int, data: PreprocessResponse) -> float:
    img = non_normalized_image(idx % data.data["real_size"], data)
    return np.mean(img)


def metadata_filename(idx: int, data: PreprocessResponse) -> str:
    return data.data['file_names'][idx]


def metadata_city(idx: int, data: PreprocessResponse) -> str:
    return data.data['cities'][idx]


def metadata_dataset(idx: int, data: PreprocessResponse) -> str:
    return data.data['dataset'][idx % data.data["real_size"]]


def metadata_idx(idx: int, data: PreprocessResponse) -> int:
    return idx


def get_metadata_json(idx: int, data: PreprocessResponse) -> Dict[str, str]:
    cloud_path = data.data['metadata'][idx]
    fpath = _download(cloud_path)
    with open(fpath, 'r') as f:
        metadata_dict = json.loads(f.read())
    return metadata_dict


def metadata_gps_heading(idx: int, data: PreprocessResponse) -> float:
    if data.data['dataset'][idx] == "cityscapes":
        return get_metadata_json(idx, data)['gpsHeading']
    else:
        return DEFAULT_GPS_HEADING


def metadata_gps_latitude(idx: int, data: PreprocessResponse) -> float:
    if data.data['dataset'][idx] == "cityscapes":
        return get_metadata_json(idx, data)['gpsLatitude']
    else:
        return DEFAULT_GPS_LATITUDE


def metadata_gps_longtitude(idx: int, data: PreprocessResponse) -> float:
    if data.data['dataset'][idx] == "cityscapes":
        return get_metadata_json(idx, data)['gpsLongitude']
    else:
        return DEFAULT_GPS_LONGTITUDE


def metadata_outside_temperature(idx: int, data: PreprocessResponse) -> float:
    if data.data['dataset'][idx] == "cityscapes":
        return get_metadata_json(idx, data)['outsideTemperature']
    else:
        return DEFAULT_TEMP


def metadata_speed(idx: int, data: PreprocessResponse) -> float:
    if data.data['dataset'][idx] == "cityscapes":
        return get_metadata_json(idx, data)['speed']
    else:
        return DEFAULT_SPEED


def metadata_yaw_rate(idx: int, data: PreprocessResponse) -> float:
    if data.data['dataset'][idx] == "cityscapes":
        return get_metadata_json(idx, data)['yawRate']
    else:
        return DEFAULT_YAW_RATE


def aug_factor_or_zero(idx: int, data: PreprocessResponse, value: float) -> float:
    if data.data["subset_name"] == "train" and AUGMENT and idx > TRAIN_SIZE - 1:
        return value.numpy()
    else:
        return 0.


def unnormalize_image(image: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return image * IMAGE_STD + IMAGE_MEAN


def image_visualizer(image: npt.NDArray[np.float32]) -> LeapImage:
    return LeapImage((unnormalize_image(image) * 255).astype(np.uint8))


def mask_visualizer(image: npt.NDArray[np.float32], mask: npt.NDArray[np.uint8]) -> LeapImageMask:
    excluded_mask = mask.sum(-1) == 0
    if len(mask.shape) > 2:
        if mask.shape[-1] == 1:
            mask = np.squeeze(mask, axis=-1)
        else:
            mask = np.argmax(mask, axis=-1)
    mask[excluded_mask] = 19
    return LeapImageMask(mask.astype(np.uint8), unnormalize_image(image).astype(np.float32), categories + ["excluded"])


def cityscape_segmentation_visualizer(mask: npt.NDArray[np.uint8]) -> LeapImage:
    if len(mask.shape) > 2:
        if mask.shape[-1] == 1:
            cat_mask = np.squeeze(mask, axis=-1)
        else:
            cat_mask = np.argmax(mask, axis=-1)  # this introduce 0 at places where no GT is present (zero all channels)
    else:
        cat_mask = mask
    cat_mask[mask.sum(-1) == 0] = 19  # this marks the place with all zeros using idx 19
    mask_image = Cityscapes.decode_target(cat_mask)
    return LeapImage(mask_image.astype(np.uint8))


jet = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
def loss_visualizer(image: npt.NDArray[np.float32], prediction: npt.NDArray[np.float32], gt: npt.NDArray[np.float32]) -> LeapImage:
    image = unnormalize_image(image)
    ls = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')
    ls_image = ls(gt, prediction).numpy()
    ls_image = ls_image.clip(0, np.percentile(ls_image, 95))
    ls_image /= ls_image.max()
    heatmap = scalarMap.to_rgba(ls_image)[..., :-1]
    # overlayed_image = ((heatmap * 0.4 + image * 0.6).clip(0,1)*255).astype(np.uint8)
    overlayed_image = ((heatmap).clip(0, 1) * 255).astype(np.uint8)
    return LeapImage(overlayed_image)


leap_binder.set_preprocess(subset_images)
leap_binder.set_input(input_image, 'normalized_image')
# leap_binder.set_input(non_normalized_image, 'image')
leap_binder.set_ground_truth(ground_truth_mask, 'mask')
leap_binder.set_metadata(metadata_background_percent, DatasetMetadataType.float, 'background_percent')
for i in range(19):  # TODO change to num classes
    leap_binder.set_metadata(metadata_percent_function_generator(i),
                             DatasetMetadataType.float,
                             Cityscapes.train_id_to_label[i] + "_" + "class_percent")
leap_binder.set_metadata(metadata_filename, DatasetMetadataType.string, 'filename')
leap_binder.set_metadata(metadata_city, DatasetMetadataType.string, 'city')
leap_binder.set_metadata(metadata_dataset, DatasetMetadataType.string, 'dataset')
leap_binder.set_metadata(metadata_idx, DatasetMetadataType.float, 'idx')
leap_binder.set_metadata(metadata_gps_heading, DatasetMetadataType.float, 'gps_heading')
leap_binder.set_metadata(metadata_gps_latitude, DatasetMetadataType.float, 'gps_latitude')
leap_binder.set_metadata(metadata_gps_longtitude, DatasetMetadataType.float, 'gps_longtitude')
leap_binder.set_metadata(metadata_outside_temperature, DatasetMetadataType.float, 'outside_temperature')
leap_binder.set_metadata(metadata_speed, DatasetMetadataType.float, 'speed')
leap_binder.set_metadata(metadata_yaw_rate, DatasetMetadataType.float, 'yaw_rate')
leap_binder.set_visualizer(image_visualizer, 'image_visualizer', LeapDataType.Image)
leap_binder.set_visualizer(mask_visualizer, 'mask_visualizer', LeapDataType.ImageMask)
leap_binder.set_visualizer(cityscape_segmentation_visualizer, 'cityscapes_visualizer', LeapDataType.Image)
leap_binder.set_visualizer(loss_visualizer, 'loss_visualizer', LeapDataType.Image)

leap_binder.add_prediction('seg_mask', categories)
# leap_binder.set_metadata(hsv_std, DatasetMetadataType.float, 'hue_std')tm_, i.fi,
