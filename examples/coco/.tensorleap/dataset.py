import os
from functools import lru_cache
from typing import List, Optional
import cv2
import numpy as np
from code_loader import dataset_binder
from code_loader.contract.enums import DatasetInputType, DatasetOutputType, DatasetMetadataType
from google.cloud import storage
from google.cloud.storage import Bucket
from pycocotools.coco import COCO
from skimage.color import gray2rgb
from skimage.io import imread
from google.auth.credentials import AnonymousCredentials
from code_loader.contract.datasetclasses import SubsetResponse
from typing import Callable, Union

BUCKET_NAME = 'example-datasets-47ml982d'
PROJECT_ID = 'example-dev-project-nmrksf0o'
image_size = 128
categories = ['person', 'car']
SUPERCATEGORY_GROUNDTRUTH = True
SUPERCATEGORY_CLASSES = ["bus", "truck", "train"]
APPLY_AUGMENTATION = True


def get_length(data):
    if data is None:
        length = None
    elif type(data) is dict and 'length' in data:
        length = data['length']
    elif type(data) is not dict:
        length = len(data)
    else:
        length = None

    return length


@lru_cache()
def _connect_to_gcs_and_return_bucket(bucket_name: str) -> Bucket:
    print("connect to GCS")
    gcs_client = storage.Client(project=PROJECT_ID, credentials=AnonymousCredentials())
    return gcs_client.bucket(bucket_name)


def _download(cloud_file_path: str, local_file_path: Optional[str] = None) -> str:
    print("download data")
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


def subset_images() -> List[SubsetResponse]:
    print("subset")

    def load_set(coco):
        # get all images containing given categories
        catIds = coco.getCatIds(categories)     # Fetch class IDs only corresponding to the filterClasses
        imgIds = coco.getImgIds(catIds=catIds)  # Get all images containing the Category IDs
        imgs = coco.loadImgs(imgIds)
        return imgs

    dataType = 'train2014'
    annFile = '{}annotations/instances_{}.json'.format("coco/ms-coco/", dataType)
    fpath = _download(annFile)
    # initialize COCO api for instance annotations
    print(fpath)
    traincoco = COCO(fpath)
    print(traincoco)
    x_train_raw = load_set(coco=traincoco)
    dataType = 'val2014'
    annFile = '{}annotations/instances_{}.json'.format("coco/ms-coco/", dataType)
    fpath = _download(annFile)
    # initialize COCO api for instance annotations
    valcoco = COCO(fpath)
    x_test_raw = load_set(coco=valcoco)
    train_size = min(len(x_train_raw), 6000)
    val_size = min(len(x_test_raw), 2800)
    supercategory_ids = traincoco.getCatIds(catNms=SUPERCATEGORY_CLASSES)
    return [
        SubsetResponse(length=train_size, data={'cocofile': traincoco, 'samples': x_train_raw[:train_size],
                                                'subdir': 'train2014', 'supercategory_ids': supercategory_ids}),
        SubsetResponse(length=val_size, data={'cocofile': valcoco, 'samples': x_test_raw[:val_size],
                                              'subdir': 'val2014', 'supercategory_ids': supercategory_ids})]


def input_image(idx, data):
    print("subset")
    data = data.data
    x = data['samples'][idx]
    filepath = "coco/ms-coco/{folder}/{file}".format(folder=data['subdir'], file=x['file_name'])
    fpath = _download(filepath)
    img = imread(fpath)
    if len(img.shape) == 2:
        # grey scale -> expand to rgb
        img = gray2rgb(img)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

    # rescale
    img = img / 255
    return img.astype(np.float)


def ground_truth_mask(idx, data):
    print("GT mask")
    data = data.data
    catIds = data['cocofile'].getCatIds(catNms=categories)
    x = data['samples'][idx]
    batch_masks = []
    annIds = data['cocofile'].getAnnIds(imgIds=x['id'], catIds=catIds, iscrowd=None)
    anns = data['cocofile'].loadAnns(annIds)
    mask = np.zeros([x['height'], x['width']])
    for ann in anns:
        _mask = data['cocofile'].annToMask(ann)
        mask[_mask > 0] = _mask[_mask > 0] * (catIds.index(ann['category_id']) + 1)
    # here we add other vehicles (truck, bus) to the car mask to create a vehicle mask
    if SUPERCATEGORY_GROUNDTRUTH:
        car_id = catIds[-1]
        other_anns_ids = data['cocofile'].getAnnIds(imgIds=x['id'], catIds=data['supercategory_ids'], iscrowd=None)
        other_anns = data['cocofile'].loadAnns(other_anns_ids)
        for j, ot_ann in enumerate(other_anns):
            _mask = data['cocofile'].annToMask(ot_ann)
            mask[_mask > 0] = _mask[_mask > 0] * (catIds.index(car_id) + 1)
    mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)[..., np.newaxis]
    return mask.astype(np.float)


def get_counts_of_instances_per_label(idx: int, data: SubsetResponse, label: str = 'all') -> int:
    data = data.data
    x = data['samples'][idx]
    catIds = data['cocofile'].getCatIds(catNms=categories)
    annIds = data['cocofile'].getAnnIds(imgIds=x['id'], catIds=catIds)
    anns_list = data['cocofile'].loadAnns(annIds)
    if label == 'all':
        return len(anns_list)
    cat_name_to_id = dict(zip(categories, catIds))
    cat_id = cat_name_to_id[label]
    cat_id_counts = {cat_id: 0 for cat_id in catIds}
    for ann in anns_list:
        cat_id_counts[ann['category_id']] += 1
    return cat_id_counts[cat_id]


def metadata_person_instances_count(idx: int, data: SubsetResponse):
    return get_counts_of_instances_per_label(idx, data, label='person')


def metadata_car_instances_count(idx: int, data: SubsetResponse):
    return get_counts_of_instances_per_label(idx, data, label='car')


def metadata_total_instances_count(idx: int, data: SubsetResponse):
    return get_counts_of_instances_per_label(idx, data, label='all')


def metadata_background_percent(idx, data):
    mask = ground_truth_mask(idx, data)
    unique, counts = np.unique(mask, return_counts=True)
    unique_per_obj = dict(zip(unique, counts))
    count_obj = unique_per_obj.get(0.0)
    if count_obj is not None:
        percent_obj = count_obj / mask.size
    else:
        percent_obj = 0.0
    return percent_obj


def metadata_person_percent(idx, data):
    mask = ground_truth_mask(idx, data)
    unique, counts = np.unique(mask, return_counts=True)
    unique_per_obj = dict(zip(unique, counts))
    count_obj = unique_per_obj.get(1.0)
    if count_obj is not None:
        percent_obj = count_obj / mask.size
    else:
        percent_obj = 0.0
    return percent_obj


def metadata_car_percent(idx, data):
    mask = ground_truth_mask(idx, data)
    unique, counts = np.unique(mask, return_counts=True)
    unique_per_obj = dict(zip(unique, counts))
    count_obj = unique_per_obj.get(2.0)
    if count_obj is not None:
        percent_obj = count_obj / mask.size
    else:
        percent_obj = 0.0
    return percent_obj


def metadata_brightness(idx: int, data: SubsetResponse):
    print("extracting metadata image brightness")
    data = data.data
    x = data['samples'][idx]
    filepath = "coco/ms-coco/{folder}/{file}".format(folder=data['subdir'], file=x['file_name'])
    fpath = _download(filepath)
    img = imread(fpath)
    if len(img.shape) == 2:
        # grascale -> expand to rgb
        img = gray2rgb(img)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

    # rescale
    img = img.astype(np.float)
    return np.mean(img)


def metadata_is_colored(idx, data):
    print("extracting metadata image brightness")
    data = data.data
    x = data['samples'][idx]
    filepath = "coco/ms-coco/{folder}/{file}".format(folder=data['subdir'], file=x['file_name'])
    fpath = _download(filepath)
    img = imread(fpath)
    is_colored = len(img.shape) > 2
    return is_colored



dataset_binder.set_subset(subset_images, 'images')

dataset_binder.set_input(input_image, 'images', DatasetInputType.Image, 'image')

dataset_binder.set_ground_truth(ground_truth_mask, 'images', ground_truth_type=DatasetOutputType.Mask, name='mask',
                                labels=['background'] + categories, masked_input="image")

dataset_binder.set_metadata(metadata_background_percent, 'images', DatasetMetadataType.float, 'background_percent')

dataset_binder.set_metadata(metadata_person_percent, 'images', DatasetMetadataType.float, 'person_percent')

dataset_binder.set_metadata(metadata_car_percent, 'images', DatasetMetadataType.float, 'car_percent')

dataset_binder.set_metadata(metadata_brightness, 'images', DatasetMetadataType.float, 'brightness')

dataset_binder.set_metadata(metadata_is_colored, 'images', DatasetMetadataType.boolean, 'is_colored')

dataset_binder.set_metadata(metadata_total_instances_count, 'images', DatasetMetadataType.int, 'total_instances_count')

dataset_binder.set_metadata(metadata_person_instances_count, 'images', DatasetMetadataType.int, 'person_instances_count')

dataset_binder.set_metadata(metadata_car_instances_count, 'images', DatasetMetadataType.int, 'car_instances_count')


