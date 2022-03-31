import os
from functools import lru_cache
from typing import List, Callable, Optional

import cv2
import numpy as np
from numpy import ndarray
from google.cloud import storage
from google.cloud.storage import Bucket
from google.auth.credentials import AnonymousCredentials
from pycocotools.coco import COCO
from skimage.color import gray2rgb
from skimage.io import imread

# Tensorleap Imports
from code_loader import dataset_binder
from code_loader.contract.datasetclasses import SubsetResponse
from code_loader.contract.enums import DatasetInputType, DatasetOutputType, DatasetMetadataType


BUCKET_NAME = 'example-datasets-47ml982d'
PROJECT_ID = 'example-dev-project-nmrksf0o'
IMAGE_SIZE = 128
CATEGORIES = ['person', 'car']

TRAIN_SIZE = 6000
TEST_SIZE = 2800

# for second experiment
SUPERCATEGORY_GROUNDTRUTH = True
SUPERCATEGORY_CLASSES = ['bus', 'truck', 'train']

# if to fetch images tha have both categories or the union of images that can include at least one category
LOAD_UNION_CATEGORIES_IMAGES = False

@lru_cache()
def _connect_to_gcs_and_return_bucket(bucket_name: str) -> Bucket:
    gcs_client = storage.Client(project=PROJECT_ID, credentials=AnonymousCredentials())
    return gcs_client.bucket(bucket_name)


def _download(cloud_file_path: str, local_file_path: Optional[str] = None) -> str:
    # if local_file_path is not specified saving in home dir
    if local_file_path is None:
        home_dir = os.getenv("HOME")
        local_file_path = os.path.join(home_dir, "Tensorleap_data", BUCKET_NAME, cloud_file_path)

    # check if file already exists
    if os.path.exists(local_file_path):
        return local_file_path

    bucket = _connect_to_gcs_and_return_bucket(BUCKET_NAME)
    dir_path = os.path.dirname(local_file_path)
    os.makedirs(dir_path, exist_ok=True)
    blob = bucket.blob(cloud_file_path)
    blob.download_to_filename(local_file_path)
    return local_file_path


# Preprocessing Function
def subset_images() -> List[SubsetResponse]:

    def load_set(coco: COCO, load_union: bool = False) -> List:
        # get all images containing given categories
        catIds = coco.getCatIds(CATEGORIES)     # Fetch class IDs only corresponding to the filterClasses
        if not load_union:
            imgIds = coco.getImgIds(catIds=catIds)  # Get all images containing the Category IDs together
        else:
            imgIds = set()
            for cat_id in catIds:
                image_ids = coco.getImgIds(catIds=[cat_id])
                imgIds.update(image_ids)
            imgIds = list(imgIds)
        imgs = coco.loadImgs(imgIds)
        return imgs

    dataType = 'train2014'
    annFile = '{}annotations/instances_{}.json'.format("coco/ms-coco/", dataType)
    fpath = _download(annFile)
    # initialize COCO api for instance annotations
    traincoco = COCO(fpath)
    x_train_raw = load_set(coco=traincoco, load_union=LOAD_UNION_CATEGORIES_IMAGES)
    dataType = 'val2014'
    annFile = '{}annotations/instances_{}.json'.format("coco/ms-coco/", dataType)
    fpath = _download(annFile)
    # initialize COCO api for instance annotations
    valcoco = COCO(fpath)
    x_test_raw = load_set(coco=valcoco, load_union=LOAD_UNION_CATEGORIES_IMAGES)
    train_size = min(len(x_train_raw), TRAIN_SIZE)
    val_size = min(len(x_test_raw), TEST_SIZE)
    supercategory_ids = traincoco.getCatIds(catNms=SUPERCATEGORY_CLASSES)
    return [
        SubsetResponse(length=train_size, data={'cocofile': traincoco, 'samples': x_train_raw[:train_size],
                                                'subdir': 'train2014', 'supercategory_ids': supercategory_ids}),
        SubsetResponse(length=val_size, data={'cocofile': valcoco, 'samples': x_test_raw[:val_size],
                                              'subdir': 'val2014', 'supercategory_ids': supercategory_ids})]


# Input Encoder
def input_image(idx: int, data: SubsetResponse) -> ndarray:
    data = data.data
    x = data['samples'][idx]
    filepath = "coco/ms-coco/{folder}/{file}".format(folder=data['subdir'], file=x['file_name'])
    fpath = _download(filepath)
    img = imread(fpath)
    if len(img.shape) == 2:
        # grey scale -> expand to rgb
        img = gray2rgb(img)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    # rescale
    img = img / 255
    return img.astype(np.float)


# Ground Truth Encoder
def ground_truth_mask(idx: int, data: SubsetResponse) -> ndarray:
    data = data.data
    catIds = data['cocofile'].getCatIds(catNms=CATEGORIES)
    x = data['samples'][idx]
    annIds = data['cocofile'].getAnnIds(imgIds=x['id'], catIds=catIds, iscrowd=None)
    anns = data['cocofile'].loadAnns(annIds)
    mask = np.zeros([x['height'], x['width']])
    for ann in anns:
        _mask = data['cocofile'].annToMask(ann)
        mask[_mask > 0] = _mask[_mask > 0] * (catIds.index(ann['category_id']) + 1)
    # here we add other vehicles (truck, bus, train) to the car mask to create a vehicle mask
    if SUPERCATEGORY_GROUNDTRUTH:
        car_id = catIds[-1]
        other_anns_ids = data['cocofile'].getAnnIds(imgIds=x['id'], catIds=data['supercategory_ids'], iscrowd=None)
        other_anns = data['cocofile'].loadAnns(other_anns_ids)
        for j, ot_ann in enumerate(other_anns):
            _mask = data['cocofile'].annToMask(ot_ann)
            mask[_mask > 0] = _mask[_mask > 0] * (catIds.index(car_id) + 1)
    mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)[..., np.newaxis]
    return mask.astype(np.float)


# Metadata functions
def get_image_percent_per_category(idx: int, data: SubsetResponse, label_key: str) -> float:
    category_id_dict = {'background': 0, 'person': 1, 'car_vehicle': 2}
    assert label_key in category_id_dict.keys()
    mask = ground_truth_mask(idx, data)
    categories, counts = np.unique(mask, return_counts=True)
    cat_value_counts = dict(zip(categories, counts))
    cat_counts = cat_value_counts.get(category_id_dict[label_key])
    return cat_counts/mask.size if cat_counts is not None else 0.0


def metadata_category_percent(label_key: str) -> Callable[[int, SubsetResponse], float]:
    def func(idx: int, data: SubsetResponse) -> float:
        return get_image_percent_per_category(idx, data, label_key=label_key)

    func.__name__ = f'metadata_{label_key}_category_percent'
    return func


def metadata_brightness(idx: int, data: SubsetResponse) -> ndarray:
    data = data.data
    x = data['samples'][idx]
    filepath = "coco/ms-coco/{folder}/{file}".format(folder=data['subdir'], file=x['file_name'])
    fpath = _download(filepath)
    img = imread(fpath)
    if len(img.shape) == 2:
        # grascale -> expand to rgb
        img = gray2rgb(img)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

    # rescale
    img = img.astype(np.float)
    return np.mean(img)


def metadata_is_colored(idx: int, data: SubsetResponse) -> bool:
    data = data.data
    x = data['samples'][idx]
    filepath = "coco/ms-coco/{folder}/{file}".format(folder=data['subdir'], file=x['file_name'])
    fpath = _download(filepath)
    img = imread(fpath)
    is_colored = len(img.shape) > 2
    return is_colored


def get_counts_of_instances_per_class(idx: int, data: SubsetResponse, label_key: str = 'all') -> int:
    data = data.data
    x = data['samples'][idx]
    all_labels = SUPERCATEGORY_CLASSES + CATEGORIES
    vehicle_labels = ['car'] + SUPERCATEGORY_CLASSES
    catIds = [data['cocofile'].getCatIds(catNms=label)[0] for label in all_labels]  # keep same labels order
    annIds = data['cocofile'].getAnnIds(imgIds=x['id'], catIds=catIds)
    anns_list = data['cocofile'].loadAnns(annIds)
    if label_key == 'all':
        return len(anns_list)   # all instances within labels
    cat_name_to_id = dict(zip(all_labels, catIds))  # map label name to its ID
    cat_id_counts = {cat_id: 0 for cat_id in catIds}    # counts dictionary
    for ann in anns_list:
        cat_id_counts[ann['category_id']] += 1
    if label_key == 'vehicle':  # count super category vehicle
        vehicle_ids = [cat_name_to_id[cat_name] for cat_name in vehicle_labels]
        return int(np.sum([cat_id_counts[cat_id] for cat_id in vehicle_ids]))
    cat_id = cat_name_to_id[label_key]
    return cat_id_counts[cat_id]


def metadata_category_instances_count(label_key: str) -> Callable[[int, SubsetResponse], int]:
    def func(idx: int, data: SubsetResponse) -> int:
        return get_counts_of_instances_per_class(idx, data, label_key=label_key)

    func.__name__ = f'metadata_{label_key}_instances_count'
    return func


def metadata_category_avg_size(label_key: str) -> Callable[[int, SubsetResponse], float]:
    def func(idx: int, data: SubsetResponse) -> float:
        percent_val = metadata_category_percent(label_key=label_key)(idx, data)
        instances_cnt = metadata_category_instances_count(label_key)(idx, data)
        return np.round(percent_val/instances_cnt, 3) if instances_cnt > 0 else 0

    func.__name__ = f'metadata_{label_key}_avg_size'
    return func


def metadata_car_vehicle_avg_size(idx: int, data: SubsetResponse) -> float:
    percent_val = metadata_category_percent(label_key='car_vehicle')(idx, data)
    label_key = 'vehicle' if SUPERCATEGORY_GROUNDTRUTH else 'car'
    instances_cnt = metadata_category_instances_count(label_key)(idx, data)
    return np.round(percent_val/instances_cnt, 3) if instances_cnt > 0 else 0


# Dataset binding functions
dataset_binder.set_subset(subset_images, 'images')

dataset_binder.set_input(input_image, 'images', DatasetInputType.Image, 'image')

dataset_binder.set_ground_truth(ground_truth_mask, 'images', ground_truth_type=DatasetOutputType.Mask, name='mask',
                                labels=['background'] + CATEGORIES, masked_input="image")

dataset_binder.set_metadata(metadata_brightness, 'images', DatasetMetadataType.float, 'brightness')

dataset_binder.set_metadata(metadata_is_colored, 'images', DatasetMetadataType.boolean, 'is_colored')

METADATA_CATEGORY_PERCENT = ['background', 'person', 'car_vehicle']  # For Super Category mode includes: car, truck, bus, train. For Category mode: only car.
for cat in METADATA_CATEGORY_PERCENT:
    dataset_binder.set_metadata(function=metadata_category_percent(cat), subset='images',
                                metadata_type=DatasetMetadataType.float, name=f'{cat}_percent')

METADATA_CATEGORY_INSTANCES_COUNT = ['all', 'person', 'car', 'bus', 'truck', 'train', 'vehicle']
for cat in METADATA_CATEGORY_INSTANCES_COUNT:
    dataset_binder.set_metadata(function=metadata_category_instances_count(cat), subset='images',
                                metadata_type=DatasetMetadataType.int, name=f'{cat}_instances_count')

METADATA_CATEGORY_AVG_SIZE = ['person']
for cat in METADATA_CATEGORY_AVG_SIZE:
    dataset_binder.set_metadata(function=metadata_category_avg_size(cat), subset='images',
                                metadata_type=DatasetMetadataType.float, name=f'{cat}_avg_size')

# For Super Category mode includes: car, truck, bus, train. For Category mode: only car.
dataset_binder.set_metadata(function=metadata_car_vehicle_avg_size, subset='images', metadata_type=DatasetMetadataType.float, name=f'car_category_avg_size')

