import numpy as np
from PIL import Image
from code_loader.contract.datasetclasses import PreprocessResponse

from armbench_segmentation import CACHE_DICTS
from armbench_segmentation.utils import polygon_to_bbox
from armbench_segmentation.yolo_helpers.yolo_utils import BOXES_GENERATOR

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
DEFAULT_BOXES = BOXES_GENERATOR.generate_anchors()
LOAD_UNION_CATEGORIES_IMAGES = True


def get_annotation_coco(idx: int, data: PreprocessResponse) -> np.ndarray:
    x = data['samples'][idx]
    coco = data['cocofile']
    # rescale
    ann_ids = coco.getAnnIds(imgIds=x['id'])
    anns = coco.loadAnns(ann_ids)
    return anns


def get_masks(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data
    MASK_SIZE = (160, 160)
    coco = data['cocofile']
    anns = get_annotation_coco(idx, data)
    masks = np.zeros([MAX_BB_PER_IMAGE, *MASK_SIZE], dtype=np.uint8)
    # mask = coco.annToMask(anns[0])
    for i in range(min(len(anns), MAX_BB_PER_IMAGE)):
        ann = anns[i]
        mask = coco.annToMask(ann)
        mask = np.array(Image.fromarray(mask).resize((MASK_SIZE[0], MASK_SIZE[1]), Image.NEAREST))
        masks[i, ...] = mask
    return masks


def get_bbs(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data
    res = CACHE_DICTS['bbs'].get(str(idx) + data['subdir'])
    if res is not None:
        return res
    x = data['samples'][idx]
    coco = data['cocofile']
    ann_ids = coco.getAnnIds(imgIds=x['id'])
    anns = coco.loadAnns(ann_ids)
    bboxes = np.zeros([MAX_BB_PER_IMAGE, 5])
    max_anns = min(MAX_BB_PER_IMAGE, len(anns))
    # mask = coco.annToMask(anns[0])
    for i in range(max_anns):
        ann = anns[i]
        img_size = (x['height'], x['width'])
        class_id = 2 - ann['category_id']
        # resize
        bbox = polygon_to_bbox(ann['segmentation'][0])
        bbox /= np.array((img_size[1], img_size[0], img_size[1], img_size[0]))
        bboxes[i, :4] = bbox
        bboxes[i, 4] = class_id
    bboxes[max_anns:, 4] = BACKGROUND_LABEL
    if len(CACHE_DICTS['bbs'].keys()) > BATCH_SIZE:
        CACHE_DICTS['bbs'] = {str(idx) + data['subdir']: bboxes}
    else:
        CACHE_DICTS['bbs'][str(idx) + data['subdir']] = bboxes
    return bboxes


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

