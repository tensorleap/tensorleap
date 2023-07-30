import numpy as np

BUCKET_NAME = 'datasets-reteai'
PROJECT_ID = 'splendid-flow-231921'
MODEL_FORMAT = "inference"
MAX_BB_PER_IMAGE = 30

IMAGE_MEAN = np.array([0.287, 0.325, 0.284])
IMAGE_STD = np.array([0.176, 0.181, 0.178])

BACKGROUND_LABEL = 19
CLASSES = 35
OFFSET = 0
CONF_THRESH = 0.35
NMS_THRESH = 0.5
OVERLAP_THRESH = 1 / 16
SMALL_BBS_TH = 0.0003  # Equivelent to ~120 pixels of area at most
STRIDES = (8, 16, 32)
image_size = (640, 640)
FEATURE_MAPS = ((80, 80), (40, 40), (20, 20))
BOX_SIZES = (((10, 13), (16, 30), (33, 23)),
             ((30, 61), (62, 45), (59, 119)),
             ((116, 90), (156, 198), (373, 326)))

## Augmentation limits
#SUPERCATEGORY_GROUNDTRUTH = False
#AUGMENT = True
# HUE_LIM = 0.3/np.pi
# SATUR_LIM = 0.3
# BRIGHT_LIM = 0.3
# CONTR_LIM = 0.3
#DEFAULT_TEMP = 19.5
#MAX_BB_PER_IMAGE = 20

#BATCH_SIZE = 32
#MAX_INSTANCES_PER_CLASS = 20

# IMAGE_MEAN = np.array([0.485, 0.456, 0.406])
# IMAGE_STD = np.array([0.229, 0.224, 0.225])