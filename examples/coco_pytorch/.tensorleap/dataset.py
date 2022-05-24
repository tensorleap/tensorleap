import os
from functools import lru_cache
from typing import List, Optional
import cv2
import numpy as np
from code_loader import leap_binder
from code_loader.contract.enums import DatasetMetadataType, Metric
from google.cloud import storage
from google.cloud.storage import Bucket
from pycocotools.coco import COCO
from skimage.color import gray2rgb
from skimage.io import imread
from google.auth.credentials import AnonymousCredentials
from code_loader.contract.datasetclasses import PreprocessResponse
from typing import Callable, Union
import tensorflow as tf
from skimage.color import rgb2hsv


BUCKET_NAME = 'example-datasets-47ml982d'
PROJECT_ID = 'example-dev-project-nmrksf0o'
image_size = 128
categories = ['person', 'car']
SUPERCATEGORY_GROUNDTRUTH = False
SUPERCATEGORY_CLASSES = ['bus', 'truck', 'train']
LOAD_UNION_CATEGORIES_IMAGES = False
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


def subset_images() -> List[PreprocessResponse]:
    print("subset")

    def load_set(coco, load_union=False):
        # get all images containing given categories
        catIds = coco.getCatIds(categories)     # Fetch class IDs only corresponding to the filterClasses
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
    print(fpath)
    traincoco = COCO(fpath)
    print(traincoco)
    x_train_raw = load_set(coco=traincoco, load_union=LOAD_UNION_CATEGORIES_IMAGES)
    dataType = 'val2014'
    annFile = '{}annotations/instances_{}.json'.format("coco/ms-coco/", dataType)
    fpath = _download(annFile)
    # initialize COCO api for instance annotations
    valcoco = COCO(fpath)
    x_test_raw = load_set(coco=valcoco, load_union=LOAD_UNION_CATEGORIES_IMAGES)
    train_size = min(len(x_train_raw), 15000)
    val_size = min(len(x_test_raw), 5000)
    supercategory_ids = traincoco.getCatIds(catNms=SUPERCATEGORY_CLASSES)
    return [
        PreprocessResponse(length=train_size, data={'cocofile': traincoco, 'samples': x_train_raw[:train_size],
                                                'subdir': 'train2014', 'supercategory_ids': supercategory_ids}),
        PreprocessResponse(length=val_size, data={'cocofile': valcoco, 'samples': x_test_raw[:val_size],
                                              'subdir': 'val2014', 'supercategory_ids': supercategory_ids})]


def input_image(idx: int, data: PreprocessResponse) -> np.ndarray:
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

def get_categorical_mask(idx: int, data: PreprocessResponse) -> np.ndarray:
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
    # here we add other vehicles (truck, bus, train) to the car mask to create a vehicle mask
    if SUPERCATEGORY_GROUNDTRUTH:
        car_id = catIds[-1]
        other_anns_ids = data['cocofile'].getAnnIds(imgIds=x['id'], catIds=data['supercategory_ids'], iscrowd=None)
        other_anns = data['cocofile'].loadAnns(other_anns_ids)
        for j, ot_ann in enumerate(other_anns):
            _mask = data['cocofile'].annToMask(ot_ann)
            mask[_mask > 0] = _mask[_mask > 0] * (catIds.index(car_id) + 1)
    mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)[..., np.newaxis]
    return mask

def ground_truth_mask(idx: int, data: PreprocessResponse) -> np.ndarray:
    mask = get_categorical_mask(idx, data)
    return tf.keras.utils.to_categorical(mask).astype(np.float)


def metadata_background_percent(idx: int, data: PreprocessResponse) -> float:
    print("extracting background percent metadata")
    mask = get_categorical_mask(idx, data)
    unique, counts = np.unique(mask, return_counts=True)
    unique_per_obj = dict(zip(unique, counts))
    count_obj = unique_per_obj.get(0.0)
    if count_obj is not None:
        percent_obj = count_obj / mask.size
    else:
        percent_obj = 0.0
    return percent_obj


def metadata_person_category_percent(idx: int, data: PreprocessResponse) -> float:
    print("extracting person percent metadata")
    mask = get_categorical_mask(idx, data)
    unique, counts = np.unique(mask, return_counts=True)
    unique_per_obj = dict(zip(unique, counts))
    count_obj = unique_per_obj.get(1.0)
    if count_obj is not None:
        percent_obj = count_obj / mask.size
    else:
        percent_obj = 0.0
    return percent_obj


def metadata_car_vehicle_category_percent(idx: int, data: PreprocessResponse) -> float:
    print("extracting car vehicle percent metadata")
    # When Super Category mode includes: car, truck, bus, train. For Category mode: only car.
    mask = get_categorical_mask(idx, data)
    unique, counts = np.unique(mask, return_counts=True)
    unique_per_obj = dict(zip(unique, counts))
    count_obj = unique_per_obj.get(2.0)
    if count_obj is not None:
        percent_obj = count_obj / mask.size
    else:
        percent_obj = 0.0
    return percent_obj


def metadata_brightness(idx: int, data: PreprocessResponse) -> float:
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


def metadata_is_colored(idx: int, data: PreprocessResponse) -> bool:
    print("extracting metadata is colored image")
    data = data.data
    x = data['samples'][idx]
    filepath = "coco/ms-coco/{folder}/{file}".format(folder=data['subdir'], file=x['file_name'])
    fpath = _download(filepath)
    img = imread(fpath)
    is_colored = len(img.shape) > 2
    return is_colored


def metadata_red_std(idx: int, data: PreprocessResponse) -> bool:
    print("extracting metadata rgb std")
    data = data.data
    x = data['samples'][idx]
    filepath = "coco/ms-coco/{folder}/{file}".format(folder=data['subdir'], file=x['file_name'])
    fpath = _download(filepath)
    img = imread(fpath)
    is_colored = len(img.shape) > 2
    return is_colored


def get_counts_of_instances_per_class(idx: int, data: PreprocessResponse, label_flag: str = 'all') -> int:
    data = data.data
    x = data['samples'][idx]
    all_labels = SUPERCATEGORY_CLASSES + categories
    vehicle_labels = ['car'] + SUPERCATEGORY_CLASSES
    catIds = [data['cocofile'].getCatIds(catNms=label)[0] for label in all_labels]  # keep same labels order
    annIds = data['cocofile'].getAnnIds(imgIds=x['id'], catIds=catIds)
    anns_list = data['cocofile'].loadAnns(annIds)
    if label_flag == 'all':
        return len(anns_list)   # all instances within labels
    cat_name_to_id = dict(zip(all_labels, catIds))  # map label name to its ID
    cat_id_counts = {cat_id: 0 for cat_id in catIds}    # counts dictionary
    for ann in anns_list:
        cat_id_counts[ann['category_id']] += 1
    if label_flag == 'vehicle':  # count super category vehicle
        vehicle_ids = [cat_name_to_id[cat_name] for cat_name in vehicle_labels]
        return np.sum([cat_id_counts[cat_id] for cat_id in vehicle_ids])
    cat_id = cat_name_to_id[label_flag]
    return cat_id_counts[cat_id]


def metadata_total_instances_count(idx: int, data: PreprocessResponse) -> int:
    print("extracting total instances metadata")
    return get_counts_of_instances_per_class(idx, data, label_flag='all')


def metadata_person_instances_count(idx: int, data: PreprocessResponse) -> int:
    print("extracting person instances metadata")
    return get_counts_of_instances_per_class(idx, data, label_flag='person')


def metadata_car_instances_count(idx: int, data: PreprocessResponse) -> int:
    print("extracting car instances metadata")
    return get_counts_of_instances_per_class(idx, data, label_flag='car')


def metadata_bus_instances_count(idx: int, data: PreprocessResponse) -> int:
    print("extracting bus instances metadata")
    return get_counts_of_instances_per_class(idx, data, label_flag='bus')


def metadata_truck_instances_count(idx: int, data: PreprocessResponse) -> int:
    print("extracting truck instances metadata")
    return get_counts_of_instances_per_class(idx, data, label_flag='truck')


def metadata_train_instances_count(idx: int, data: PreprocessResponse) -> int:
    print("extracting train instances metadata")
    return get_counts_of_instances_per_class(idx, data, label_flag='train')


def metadata_vehicle_instances_count(idx: int, data: PreprocessResponse) -> int:
    print("extracting vehicle instances metadata")
    return get_counts_of_instances_per_class(idx, data, label_flag='vehicle')


def metadata_person_category_avg_size(idx: int, data: PreprocessResponse) -> float:
    print("extracting person average size metadata")
    percent_val = metadata_person_category_percent(idx, data)
    instances_cnt = metadata_person_instances_count(idx, data)
    return np.round(percent_val/instances_cnt, 3) if instances_cnt > 0 else 0


def metadata_car_vehicle_category_avg_size(idx: int, data: PreprocessResponse) -> float:
    print("extracting car or vehicle average size metadata")
    percent_val = metadata_car_vehicle_category_percent(idx, data)
    if SUPERCATEGORY_GROUNDTRUTH:
        instances_cnt = metadata_vehicle_instances_count(idx, data)
    else:
        instances_cnt = metadata_car_instances_count(idx, data)
    return np.round(percent_val/instances_cnt, 3) if instances_cnt > 0 else 0


def hsv_std(idx: int, data: PreprocessResponse) -> float:
    hsv_image = rgb2hsv(input_image(idx, data))
    hue = hsv_image[...,0]
    return hue.std()


leap_binder.set_preprocess(subset_images)
leap_binder.set_input(input_image, 'images')
leap_binder.set_ground_truth(ground_truth_mask, 'mask')
leap_binder.set_metadata(metadata_background_percent,  DatasetMetadataType.float, 'background_percent')
leap_binder.set_metadata(metadata_person_category_percent,  DatasetMetadataType.float, 'person_percent')
leap_binder.set_metadata(metadata_car_vehicle_category_percent, DatasetMetadataType.float, 'car_percent')
leap_binder.set_metadata(metadata_brightness,  DatasetMetadataType.float, 'brightness')
leap_binder.set_metadata(metadata_is_colored,  DatasetMetadataType.boolean, 'is_colored')
leap_binder.set_metadata(metadata_total_instances_count,  DatasetMetadataType.int, 'total_instances_count')
leap_binder.set_metadata(metadata_person_instances_count,  DatasetMetadataType.int, 'person_instances_count')
leap_binder.set_metadata(metadata_car_instances_count,  DatasetMetadataType.int, 'car_instances_count')
leap_binder.set_metadata(metadata_bus_instances_count,  DatasetMetadataType.int, 'bus_instances_count')
leap_binder.set_metadata(metadata_truck_instances_count,  DatasetMetadataType.int, 'truck_instances_count')
leap_binder.set_metadata(metadata_train_instances_count,  DatasetMetadataType.int, 'train_instances_count')
leap_binder.set_metadata(metadata_vehicle_instances_count,  DatasetMetadataType.int, 'vehicle_instances_count')
leap_binder.set_metadata(metadata_person_category_avg_size,  DatasetMetadataType.float, 'person_avg_size')
leap_binder.set_metadata(metadata_car_vehicle_category_avg_size, DatasetMetadataType.float, 'car_vehicle_avg_size')
leap_binder.add_prediction('seg_mask', ['background'] + categories, [Metric.MeanIOU])
leap_binder.set_metadata(hsv_std, DatasetMetadataType.float, 'hue_std')
