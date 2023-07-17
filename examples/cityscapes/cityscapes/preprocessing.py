BATCH_SIZE = 32

BUCKET_NAME = 'datasets-reteai'
PROJECT_ID =
PROJECT_ID = 'splendid-flow-231921'
DIR = "Cityscapes/gtFine_trainvaltest"
IMG_FOLDER = "gtFine"

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
LOAD_UNION_CATEGORIES_IMAGES = True


def load_set(coco, load_union=False):
    # get all images containing given categories
    CATEGORIES = []
    catIds = coco.getCatIds(CATEGORIES)  # Fetch class IDs only corresponding to the Classes
    if not load_union:
        imgIds = coco.getImgIds(catIds=catIds)  # Get all images containing the Category IDs together
    else:  # get images contains any of the classes
        imgIds = set()
        for cat_id in catIds:
            image_ids = coco.getImgIds(catIds=[cat_id])
            imgIds.update(image_ids)
        imgIds = list(imgIds)[:-1]  # we're missing the last image for some reason
    imgs = coco.loadImgs(imgIds)
    return imgs
