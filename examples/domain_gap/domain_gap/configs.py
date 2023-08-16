import numpy as np

# --------------- GCS --------------------
BUCKET_NAME = 'datasets-reteai'
PROJECT_ID = 'splendid-flow-231921'

# --------------- Data --------------------
NORM_CS = False
SEED = 42
NUM_CLASSES = 19
IMAGE_SIZE = (2048, 1024)  # TODO check all occurences and fix
TRAIN_SIZE, VAL_SIZE = 400, 20
TRAIN_PERCENT = 0.8

SUPERCATEGORY_GROUNDTRUTH = False

LOAD_UNION_CATEGORIES_IMAGES = False

# --------------- Augmentations --------------------
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