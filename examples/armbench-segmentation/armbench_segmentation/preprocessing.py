BATCH_SIZE = 32

BUCKET_NAME = 'datasets-reteai'
PROJECT_ID = 'splendid-flow-231921'
DIR = "Amazon/armbench-segmentation-0.1/armbench-segmentation-0.1"
IMG_FOLDER = "mix-object-tote"

TRAIN_SIZE = 1000
VAL_SIZE = 1000
# TEST_SIZE = 1000
UL_SIZE = 1000

CATEGORIES = ['Object', 'Tote']  # class names
MAX_INSTANCES_PER_CLASS = 20
INSTANCES = [f"{c}_{i + 1}" for c in CATEGORIES for i in range(MAX_INSTANCES_PER_CLASS)]

IMAGE_SIZE = (640, 640)
BACKGROUND_LABEL = 2
MODEL_FORMAT = "inference"
MAX_BB_PER_IMAGE = 20
CLASSES = 2
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
