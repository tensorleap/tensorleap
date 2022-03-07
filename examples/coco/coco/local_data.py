import os
from typing import Optional, List
from functools import lru_cache
from google.cloud import storage
from google.cloud.storage import Bucket
from google.auth.credentials import AnonymousCredentials
from pycocotools.coco import COCO

BUCKET_NAME = 'example-datasets-47ml982d'
PROJECT_ID = 'example-dev-project-nmrksf0o'
image_size = 128
categories = ['person', 'car']
SUPERCATEGORY_GROUNDTRUTH = True
SUPERCATEGORY_CLASSES = ["bus", "truck", "train"]
APPLY_AUGMENTATION = True


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


def subset_images():

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


subset_images()

