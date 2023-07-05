import copy
from functools import lru_cache
from typing import List, Optional, Dict
from code_loader import leap_binder
from code_loader.contract.enums import DatasetMetadataType, Metric
from google.cloud import storage
from google.cloud.storage import Bucket
from code_loader.contract.datasetclasses import PreprocessResponse
import tensorflow as tf
from tensorflow.python.ops.stateless_random_ops import stateless_random_uniform
import os
from collections import namedtuple
from PIL import Image
from google.oauth2 import service_account
import numpy as np
import numpy.typing as npt
import json
from pathlib import Path
from matplotlib import colors
from matplotlib import cm as cmx
import matplotlib.pyplot as plt
from code_loader.contract.visualizer_classes import LeapImage, LeapImageMask
from code_loader.contract.enums import (
    LeapDataType
)


class Cityscapes:
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.

    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        CityscapesClass('unlabeled', 0, 19, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 19, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 19, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 19, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 19, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 19, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 19, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 19, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 19, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 19, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 19, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 19, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 19, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 19, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 19, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, 19, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    train_id_to_color = np.array(list({cls.train_id: cls.color for cls in classes[::-1]}.values())[::-1])
    id_to_train_id = np.array([c.train_id for c in classes])
    train_id_to_label = {label.train_id: label.name for label in classes}

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def encode_target_cityscapes(cls, target):
        target[target == 255] = 19
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        return cls.train_id_to_color[target]


BUCKET_NAME = 'datasets-reteai'
PROJECT_ID = 'splendid-flow-231921'
image_size = (2048, 1024)  # TODO check all occurences and fix
train_size, val_size = 400, 90
subset_sizes = [train_size, val_size]
categories = [Cityscapes.classes[i].name for i in range(len(Cityscapes.classes)) if Cityscapes.classes[i].train_id < 19]
SUPERCATEGORY_GROUNDTRUTH = False
SUPERCATEGORY_CLASSES = np.unique([Cityscapes.classes[i].category for i in range(len(Cityscapes.classes)) if
                                   Cityscapes.classes[i].train_id < 19])
LOAD_UNION_CATEGORIES_IMAGES = False
APPLY_AUGMENTATION = True
IMAGE_MEAN = np.array([0.485, 0.456, 0.406])
IMAGE_STD = np.array([0.229, 0.224, 0.225])
KITTI_MEAN = np.array([0.379, 0.398, 0.384])
KITTI_STD = np.array([0.298, 0.308, 0.315])
CITYSCAPES_MEAN = np.array([0.287, 0.325, 0.284])
CITYSCAPES_STD = np.array([0.176, 0.181, 0.178])
VAL_INDICES = [190, 198, 45, 25, 141, 104, 17, 162, 49, 167, 168, 34, 150, 113, 44,
               182, 196, 11, 6, 46, 133, 74, 81, 65, 66, 79, 96, 92, 178, 103]
AUGMENT = True
SUBSET_REPEATS = [1, 1]
# Augmentation limits
HUE_LIM = 0.3 / np.pi
SATUR_LIM = 0.3
BRIGHT_LIM = 0.3
CONTR_LIM = 0.3
DEFAULT_GPS_HEADING = 281.
DEFAULT_GPS_LATITUDE = 50.780881831805594
DEFAULT_GPS_LONGTITUDE = 6.108147476339736
DEFAULT_TEMP = 19.5
DEFAULT_SPEED = 10.81
DEFAULT_YAW_RATE = 0.171


@lru_cache()
def _connect_to_gcs_and_return_bucket(bucket_name: str) -> Bucket:
    auth_secret_string = os.environ['AUTH_SECRET']
    auth_secret = json.loads(auth_secret_string)
    if type(auth_secret) is dict:
        # getting credentials from dictionary account info
        credentials = service_account.Credentials.from_service_account_info(auth_secret)
    else:
        # getting credentials from path
        credentials = service_account.Credentials.from_service_account_file(auth_secret)
    project = credentials.project_id
    gcs_client = storage.Client(project=project, credentials=credentials)
    return gcs_client.bucket(bucket_name)


def _download(cloud_file_path: str, local_file_path: Optional[str] = None) -> str:
    # if local_file_path is not specified saving in home dir
    if local_file_path is None:
        home_dir = os.getenv("HOME")
        local_file_path = os.path.join(home_dir, "Tensorleap", BUCKET_NAME, cloud_file_path)

    # check if file already exists
    if os.path.exists(local_file_path):
        return local_file_path
    bucket = _connect_to_gcs_and_return_bucket(BUCKET_NAME)
    dir_path = os.path.dirname(local_file_path)
    os.makedirs(dir_path, exist_ok=True)
    blob = bucket.blob(cloud_file_path)
    blob.download_to_filename(local_file_path)
    return local_file_path


def get_cityscapes_data() -> List[PreprocessResponse]:
    np.random.seed(42)
    bucket = _connect_to_gcs_and_return_bucket(BUCKET_NAME)
    dataset_path = Path('Cityscapes')
    responses = []
    TRAIN_PERCENT = 0.8
    FOLDERS_NAME = ["zurich", "weimar", "ulm", "tubingen", "stuttgart", "strasbourg", "monchengladbach", "krefeld",
                    "jena",
                    "hanover", "hamburg", "erfurt", "dusseldorf", "darmstadt", "cologne", "bremen", "bochum", "aachen"]
    FOLDERS_NAME = [FOLDERS_NAME[-1], FOLDERS_NAME[0]]
    all_images = [[], []]
    all_gt_images = [[], []]
    all_gt_labels = [[], []]
    all_file_names = [[], []]
    all_cities = [[], []]
    all_metadata = [[], []]
    for folder_name in FOLDERS_NAME:
        image_list = [obj.name for obj in bucket.list_blobs(
            prefix=str(dataset_path / "leftImg8bit_trainvaltest/leftImg8bit/train" / folder_name))]
        permuted_list = np.random.permutation(image_list)
        file_names = ["_".join(os.path.basename(pth).split("_")[:-1]) for pth in permuted_list]
        images = [
            str(dataset_path / "leftImg8bit_trainvaltest/leftImg8bit/train" / folder_name / fn) + "_leftImg8bit.png" for
            fn in file_names]
        gt_labels = [str(dataset_path / "gtFine_trainvaltest/gtFine/train" / folder_name / fn) + "_gtFine_labelIds.png"
                     for fn in file_names]
        gt_images = [str(dataset_path / "gtFine_trainvaltest/gtFine/train" / folder_name / fn) + "_gtFine_color.png" for
                     fn in file_names]
        metadata_json = [str(dataset_path / "vehicle_trainvaltest/vehicle/train" / folder_name / fn) + "_vehicle.json"
                         for fn in file_names]
        train_size = int(len(permuted_list) * TRAIN_PERCENT)
        all_images[0], all_images[1] = all_images[0] + images[:train_size], all_images[1] + images[train_size:]
        all_gt_images[0], all_gt_images[1] = all_gt_images[0] + gt_images[:train_size], all_gt_images[1] + gt_images[
                                                                                                           train_size:]
        all_gt_labels[0], all_gt_labels[1] = all_gt_labels[0] + gt_labels[:train_size], all_gt_labels[1] + gt_labels[
                                                                                                           train_size:]
        all_file_names[0], all_file_names[1] = all_file_names[0] + file_names[:train_size], all_file_names[
            1] + file_names[train_size:]
        all_cities[0], all_cities[1] = all_cities[0] + [folder_name] * train_size, all_cities[1] + [folder_name] * (
                    len(permuted_list) - train_size)
        all_metadata[0], all_metadata[1] = all_metadata[0] + metadata_json[:train_size], all_metadata[
            1] + metadata_json[train_size:]
    responses = [PreprocessResponse(length=len(all_images[0]), data={
        "image_path": all_images[0],
        "subset_name": "train",
        "gt_path": all_gt_labels[0],
        "gt_image_path": all_gt_images[0],
        "real_size": len(all_images[0]),
        "file_names": all_file_names[0],
        "cities": all_cities[0],
        "metadata": all_metadata[0],
        "dataset": ["cityscapes"] * len(all_images[0])}),
                 PreprocessResponse(length=len(all_images[1]), data={
                     "image_path": all_images[1],
                     "subset_name": "val",
                     "gt_path": all_gt_labels[1],
                     "gt_image_path": all_gt_images[1],
                     "real_size": len(all_images[1]),
                     "file_names": all_file_names[1],
                     "cities": all_cities[1],
                     "metadata": all_metadata[1],
                     "dataset": ["cityscapes"] * len(all_images[1])})]
    return responses


def get_kitti_data() -> Dict[str, List[str]]:
    dataset_path = Path('KITTI/data_semantics/training')
    responses = []
    train_indices = [i for i in range(200) if i not in VAL_INDICES]
    indices_lists = [train_indices, VAL_INDICES]
    data_dict = {'train': {}, "validation": {}}
    TRAIN_SIZE = 170
    VAL_SIZE = 30
    for indices, size, title in zip(indices_lists, (TRAIN_SIZE, VAL_SIZE), ("train", "validation")):
        images = [str(dataset_path / "image_2" / (str(i).zfill(6) + "_10.png")) for i in indices]
        gt_labels = [str(dataset_path / "semantic" / (str(i).zfill(6) + "_10.png")) for i in indices]
        gt_images = [str(dataset_path / "semantic_rgb" / (str(i).zfill(6) + "_10.png")) for i in indices]
        data_dict[title]['image_path'] = images
        data_dict[title]['gt_path'] = gt_labels
        data_dict[title]['gt_image_path'] = gt_images
    return data_dict


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
    return normalized_image.astype(np.float)


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
    return tf.keras.utils.to_categorical(mask, num_classes=20).astype(np.float)[...,
           :19]  # Remove background class from cross-entropy


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


def loss_visualizer(image: npt.NDArray[np.float32], prediction: npt.NDArray[np.float32], gt: npt.NDArray[np.float32]):
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
