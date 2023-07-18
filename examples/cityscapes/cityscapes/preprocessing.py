import os
from typing import List
import numpy as np
from collections import namedtuple
from pathlib import Path

from cityscapes.gcs_utils import _connect_to_gcs_and_return_bucket

from code_loader.contract.datasetclasses import PreprocessResponse

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
    def get_class_id(cls, class_name):
        for class_ in cls.classes:
            if class_.name == class_name:
                return class_.id
        return None

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
image_size = (2048, 1024) #TODO check all occurences and fix
categories = [Cityscapes.classes[i].name for i in range(len(Cityscapes.classes)) if Cityscapes.classes[i].train_id < 19]
SUPERCATEGORY_GROUNDTRUTH = False
SUPERCATEGORY_CLASSES = np.unique([Cityscapes.classes[i].category for i in range(len(Cityscapes.classes)) if
                                   Cityscapes.classes[i].train_id < 19])
LOAD_UNION_CATEGORIES_IMAGES = False
APPLY_AUGMENTATION = True
IMAGE_MEAN = np.array([0.485, 0.456, 0.406])
IMAGE_STD = np.array([0.229, 0.224, 0.225])
VAL_INDICES = [190, 198, 45, 25, 141, 104, 17, 162, 49, 167, 168, 34, 150, 113, 44,
               182, 196, 11, 6, 46, 133, 74, 81, 65, 66, 79, 96, 92, 178, 103]
AUGMENT = True
SUBSET_REPEATS = [1,1]
# Augmentation limits
HUE_LIM = 0.3/np.pi
SATUR_LIM = 0.3
BRIGHT_LIM = 0.3
CONTR_LIM = 0.3
DEFAULT_GPS_HEADING = 281.
DEFAULT_GPS_LATITUDE = 50.780881831805594
DEFAULT_GPS_LONGTITUDE = 6.108147476339736
DEFAULT_TEMP = 19.5
DEFAULT_SPEED = 10.81
DEFAULT_YAW_RATE = 0.171

BACKGROUND_LABEL = 19
MODEL_FORMAT = "inference"
MAX_BB_PER_IMAGE = 20
CLASSES = 35
FEATURE_MAPS = ((80, 80), (40, 40), (20, 20))
BOX_SIZES = (((10, 13), (16, 30), (33, 23)),
             ((30, 61), (62, 45), (59, 119)),
             ((116, 90), (156, 198), (373, 326)))
OFFSET = 0
STRIDES = (8, 16, 32)
CONF_THRESH = 0.35
NMS_THRESH = 0.5
OVERLAP_THRESH = 1 / 16
SMALL_BBS_TH = 0.0003  # Equivelent to ~120 pixels of area at most
LOAD_UNION_CATEGORIES_IMAGES = True
BATCH_SIZE = 32
MAX_INSTANCES_PER_CLASS = 20


def load_cityscapes_data():
    np.random.seed(42)
    bucket = _connect_to_gcs_and_return_bucket(BUCKET_NAME)
    dataset_path = Path('Cityscapes')
    responses = []
    TRAIN_PERCENT = 0.8
    FOLDERS_NAME = ["zurich", "weimar", "ulm", "tubingen", "stuttgart", "strasbourg", "monchengladbach", "krefeld", "jena",
                    "hanover", "hamburg", "erfurt", "dusseldorf", "darmstadt", "cologne", "bremen", "bochum", "aachen"]
    FOLDERS_NAME = [FOLDERS_NAME[-1], FOLDERS_NAME[0]]
    all_images = [[], []]
    all_gt_images = [[], []]
    all_gt_labels = [[], []]
    all_gt_labels_for_bbx = [[], []]
    all_file_names = [[], []]
    all_cities = [[], []]
    all_metadata = [[], []]
    for folder_name in FOLDERS_NAME:
        image_list = [obj.name for obj in bucket.list_blobs(prefix=str(dataset_path / "leftImg8bit_trainvaltest/leftImg8bit/train" / folder_name))]
        permuted_list = np.random.permutation(image_list)
        file_names = ["_".join(os.path.basename(pth).split("_")[:-1]) for pth in permuted_list]
        images = [str(dataset_path / "leftImg8bit_trainvaltest/leftImg8bit/train" / folder_name / fn) + "_leftImg8bit.png" for fn in file_names]
        gt_images = [str(dataset_path / "gtFine_trainvaltest/gtFine/train" / folder_name / fn) + "_gtFine_color.png" for fn in file_names]
        gt_labels = [str(dataset_path / "gtFine_trainvaltest/gtFine/train" / folder_name / fn) + "_gtFine_labelIds.png" for fn in file_names]
        gt_labels_for_bbx = [str(dataset_path / "gtFine_trainvaltest/gtFine/train" / folder_name / fn) + "_gtFine_polygons.json" for fn in file_names]
        metadata_json = [str(dataset_path / "vehicle_trainvaltest/vehicle/train" / folder_name / fn) + "_vehicle.json" for fn in file_names] #more mets data on images
        train_size = int(len(permuted_list)*TRAIN_PERCENT)

        all_images[0], all_images[1] = all_images[0] + images[:train_size], all_images[1] + images[train_size:]
        all_gt_images[0], all_gt_images[1] = all_gt_images[0] + gt_images[:train_size], all_gt_images[1] + gt_images[train_size:]
        all_gt_labels[0], all_gt_labels[1] = all_gt_labels[0] + gt_labels[:train_size], all_gt_labels[1] + gt_labels[train_size:]
        all_gt_labels_for_bbx[0], all_gt_labels_for_bbx[1] = all_gt_labels_for_bbx[0] + gt_labels_for_bbx[:train_size], all_gt_labels_for_bbx[1] + gt_labels_for_bbx[train_size:]
        all_file_names[0], all_file_names[1] = all_file_names[0] + file_names[:train_size], all_file_names[1] + file_names[train_size:]
        all_metadata[0], all_metadata[1] = all_metadata[0] + metadata_json[:train_size], all_metadata[1] + metadata_json[train_size:]
        all_cities[0], all_cities[1] = all_cities[0] + [folder_name]*train_size, all_cities[1] + [folder_name]*(len(permuted_list)-train_size)

    return all_images, all_gt_images, all_gt_labels, all_gt_labels_for_bbx, all_file_names, all_metadata, all_cities

