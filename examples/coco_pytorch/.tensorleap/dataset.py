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
from typing import Callable
import tensorflow as tf

from skimage.color import rgb2hsv

BUCKET_NAME = 'example-datasets-47ml982d'
PROJECT_ID = 'example-dev-project-nmrksf0o'

TRAIN_SIZE = 15000
TEST_SIZE = 5000

IMAGE_SIZE = 256

# preprocess based on pre-trained DeepLab
preprocess = tf.keras.layers.Normalization(
    axis=-1, mean=[0.485, 0.456, 0.406], variance=[(0.229) ** 2, (0.224) ** 2, (0.225) ** 2]
)

CATEGORIES = [
    # "background",
    "airplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "dining table",
    "dog",
    "horse",
    "motorcycle",
    "person",
    "potted plant",
    "sheep",
    "couch",
    "train",
    "tv"
]

LOAD_UNION_CATEGORIES_IMAGES = True
APPLY_AUGMENTATION = True

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


def subset_images() -> List[PreprocessResponse]:
    def load_set(coco, load_union=False):
        # get all images containing given categories
        catIds = coco.getCatIds(CATEGORIES)  # Fetch class IDs only corresponding to the Classes
        if not load_union:
            imgIds = coco.getImgIds(catIds=catIds)  # Get all images containing the Category IDs together
        else:  # get images contains any of the classes
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
    return [
        PreprocessResponse(length=train_size, data={'cocofile': traincoco, 'samples': x_train_raw[:train_size],
                                                    'subdir': 'train2014'}),
        PreprocessResponse(length=val_size, data={'cocofile': valcoco, 'samples': x_test_raw[:val_size],
                                                  'subdir': 'val2014'})]


def input_image(idx: int, data: PreprocessResponse) -> np.ndarray:
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
    # normalize
    img = preprocess(img)
    img = img.numpy()
    return img.astype(np.float)


def get_categorical_mask(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data
    catIds = data['cocofile'].getCatIds(catNms=CATEGORIES)
    x = data['samples'][idx]
    batch_masks = []
    annIds = data['cocofile'].getAnnIds(imgIds=x['id'], catIds=catIds, iscrowd=None)
    anns = data['cocofile'].loadAnns(annIds)
    mask = np.zeros((x['height'], x['width'], len(CATEGORIES) + 1))
    for ann in anns:
        _mask = data['cocofile'].annToMask(ann)
        mask[_mask > 0, (catIds.index(ann['category_id']) + 1)] = _mask[_mask > 0]
    mask[np.sum(mask, axis=2) == 0, 0] = 1  # encode background
    mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)  # [..., np.newaxis]
    return mask


def ground_truth_mask(idx: int, data: PreprocessResponse) -> np.ndarray:
    mask = get_categorical_mask(idx, data)
    return mask

def sample_id(idx: int, data: PreprocessResponse) -> int:
    return idx

def metadata_red_std(idx: int, data: PreprocessResponse) -> bool:
    data = data.data
    x = data['samples'][idx]
    filepath = "coco/ms-coco/{folder}/{file}".format(folder=data['subdir'], file=x['file_name'])
    fpath = _download(filepath)
    img = imread(fpath)
    is_colored = len(img.shape) > 2
    return is_colored



def metadata_brightness(idx: int, data: PreprocessResponse) -> float:
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


def metadata_is_colored(idx: int, data: PreprocessResponse) -> bool:
    data = data.data
    x = data['samples'][idx]
    filepath = "coco/ms-coco/{folder}/{file}".format(folder=data['subdir'], file=x['file_name'])
    fpath = _download(filepath)
    img = imread(fpath)
    is_colored = len(img.shape) > 2
    return is_colored


def metadata_red_std(idx: int, data: PreprocessResponse) -> bool:
    data = data.data
    x = data['samples'][idx]
    filepath = "coco/ms-coco/{folder}/{file}".format(folder=data['subdir'], file=x['file_name'])
    fpath = _download(filepath)
    img = imread(fpath)
    is_colored = len(img.shape) > 2
    return is_colored


def hsv_std(idx: int, data: PreprocessResponse) -> float:
    hsv_image = rgb2hsv(input_image(idx, data))
    hue = hsv_image[..., 0]
    return hue.std()


def get_category_instances_count(idx: int, data: PreprocessResponse, label_key: str = 'all') -> int:
    data = data.data
    x = data['samples'][idx]
    catIds = [data['cocofile'].getCatIds(catNms=label)[0] for label in CATEGORIES]  # keep same labels order
    annIds = data['cocofile'].getAnnIds(imgIds=x['id'], catIds=catIds)
    anns_list = data['cocofile'].loadAnns(annIds)
    if label_key == 'all':
        return len(anns_list)  # all instances within labels
    cat_to_id = dict(zip(CATEGORIES, catIds))  # map label name to its ID
    cat_id_counts = {cat_id: 0 for cat_id in catIds}  # counts dictionary
    for ann in anns_list:
        cat_id_counts[ann['category_id']] += 1
    return sum(cat_id_counts.values()) if label_key == 'all' else cat_id_counts[cat_to_id[label_key]]


def metadata_category_instances_count(label_key: str) -> Callable[[int, PreprocessResponse], int]:
    def func(idx: int, data: PreprocessResponse) -> int:
        return get_category_instances_count(idx, data, label_key=label_key)

    func.__name__ = f'metadata_{label_key}_instances_count'
    return func


def pixel_accuracy(y_true, y_pred):
    from tensorflow.python.ops import math_ops
    from tensorflow.python.keras import backend
    # per_pixel_accuracy = categorical_accuracy(y_true, y_pred)
    per_pixel_accuracy = math_ops.cast(
        math_ops.equal(
            math_ops.argmax(y_true, axis=-1), math_ops.argmax(y_pred, axis=-1)),
        backend.floatx())
    accuracy = tf.reduce_sum(per_pixel_accuracy, axis=[1, 2])
    return accuracy


leap_binder.set_preprocess(subset_images)
leap_binder.set_input(input_image, 'images')
leap_binder.set_ground_truth(ground_truth_mask, 'mask')
leap_binder.add_prediction('seg_mask', ['background'] + CATEGORIES, [Metric.MeanIOU], [pixel_accuracy])

leap_binder.set_metadata(sample_id, DatasetMetadataType.int, 'sample_id')
leap_binder.set_metadata(metadata_brightness, DatasetMetadataType.float, 'brightness')
leap_binder.set_metadata(metadata_is_colored, DatasetMetadataType.boolean, 'is_colored')
leap_binder.set_metadata(hsv_std, DatasetMetadataType.float, 'hue_std')

metadata_cats_instances_cnt = CATEGORIES + ['all']
for cat in metadata_cats_instances_cnt:
    leap_binder.set_metadata(function=metadata_category_instances_count(cat),
                             metadata_type=DatasetMetadataType.int, name=f'{cat}_instances_count')


