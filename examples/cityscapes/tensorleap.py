import os
from typing import Union, List

import numpy as np
import tensorflow as tf
from PIL import Image
from datasets import load_dataset

from code_loader import leap_binder
from code_loader.contract.enums import (
    DatasetMetadataType,
    LeapDataType
)

from code_loader.contract.datasetclasses import PreprocessResponse
from pycocotools.coco import COCO

from cityscapes.gcs_utils import _download
from cityscapes.preprocessing import DIR, IMG_FOLDER, load_set, LOAD_UNION_CATEGORIES_IMAGES, TRAIN_SIZE, VAL_SIZE


# ----------------------------------------------------data processing--------------------------------------------------
def subset_images():
    ann_file = os.path.join(DIR, IMG_FOLDER, 'train/zurich/zurich_000000_000019_gtFine_color.png')
    fpath = _download(ann_file) #TODO: this have only one image. look at dani project
    # initialize COCO api for instance annotations
    train = COCO(fpath)
    x_train_raw = load_set(coco=train, load_union=LOAD_UNION_CATEGORIES_IMAGES)

    ann_file = os.path.join(DIR, IMG_FOLDER, "test.json")
    fpath = _download(ann_file)
    val = COCO(fpath)
    x_val_raw = load_set(coco=val, load_union=LOAD_UNION_CATEGORIES_IMAGES)

    train_size = min(len(x_train_raw), TRAIN_SIZE)
    val_size = min(len(x_val_raw), VAL_SIZE)
    np.random.seed(0)
    train_idx, val_idx = np.random.choice(len(x_train_raw), train_size), np.random.choice(len(x_val_raw), val_size)
    return [PreprocessResponse(length=train_size, data={'cocofile': train,
                                                        'samples': np.take(x_train_raw, train_idx),
                                                        'subdir': 'train'}),
            PreprocessResponse(length=val_size, data={'cocofile': val,
                                                      'samples': np.take(x_val_raw, val_idx),
                                                      'subdir': 'test'})]

if __name__=='__main__':

    dataset = load_dataset("huggan/cityscapes")
    subset_images()