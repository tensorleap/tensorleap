import os
import json
from copy import deepcopy
from functools import lru_cache
from typing import List, Optional, Tuple, Union

from google.cloud import storage
from google.cloud.storage import Bucket
from google.oauth2 import service_account

from pycocotools.coco import COCO
import tensorflow as tf
from PIL import Image
import numpy as np
from numpy.typing import NDArray

from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader import leap_binder
from code_loader.helpers.detection.yolo.utils import reshape_output_list
from code_loader.helpers.detection.yolo.loss import YoloLoss
from code_loader.helpers.detection.yolo.grid import Grid
from code_loader.helpers.detection.yolo.decoder import Decoder
from code_loader.contract.visualizer_classes import LeapImageWithBBox, LeapImageMask
from code_loader.contract.responsedataclasses import BoundingBox
from code_loader.helpers.detection.utils import xyxy_to_xywh_format, xywh_to_xyxy_format
from code_loader.contract.enums import (
    DatasetMetadataType,
    LeapDataType
)

from armbench_segmentation import CACHE_DICTS

cache_dicts = CACHE_DICTS

BATCH_SIZE = 8

BUCKET_NAME = 'datasets-reteai'
PROJECT_ID = 'splendid-flow-231921'
DIR = "Amazon/armbench-segmentation-0.1/armbench-segmentation-0.1"
IMG_FOLDER = "mix-object-tote"

TRAIN_SIZE = 1000
VAL_SIZE = 1000
# TEST_SIZE = 1000
UL_SIZE = 1000

CATEGORIES = ['Object', 'Tote']  # class names # class names
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
OVERLAP_THRESH = 1 / 16  # might need to be 1/16
SMALL_BBS_TH = 0.0003  # Equivelent to ~120 pixels of area at most
BOXES_GENERATOR = Grid(image_size=IMAGE_SIZE, feature_maps=FEATURE_MAPS, box_sizes=BOX_SIZES,
                       strides=STRIDES, offset=OFFSET)
DEFAULT_BOXES = BOXES_GENERATOR.generate_anchors()

LOSS_FN = YoloLoss(num_classes=CLASSES, overlap_thresh=OVERLAP_THRESH,
                   features=FEATURE_MAPS, anchors=np.array(BOX_SIZES),
                   default_boxes=DEFAULT_BOXES, background_label=BACKGROUND_LABEL,
                   from_logits=False if MODEL_FORMAT == "inference" else True,
                   image_size=IMAGE_SIZE,
                   yolo_match=True,
                   semantic_instance=True)

DECODER = Decoder(CLASSES,
                  background_label=BACKGROUND_LABEL,
                  top_k=50,
                  conf_thresh=CONF_THRESH,
                  nms_thresh=NMS_THRESH,
                  semantic_instance=True,
                  max_bb=50,
                  max_bb_per_layer=50)

# IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
# IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation
# preprocess based on pre-trained Yolov5
# preprocess = tf.keras.layers.Normalization(
#     axis=-1, mean=IMAGENET_MEAN, variance=np.power(IMAGENET_STD, 2)
# )

LOAD_UNION_CATEGORIES_IMAGES = True


@lru_cache()
def _connect_to_gcs_and_return_bucket(bucket_name: str) -> Bucket:
    auth_secret_string = os.environ['AUTH_SECRET']
    auth_secret = json.loads(auth_secret_string)
    if type(auth_secret) is dict:
        # getting credentials from dictionary account info
        credentials = service_account.Credentials.from_service_account_info(auth_secret)
    else:
        # getting credentials from path
        credentials = service_account.Credentials.from_service_account_file(auth_secret)
    # print("connect to GCS")
    gcs_client = storage.Client(project=PROJECT_ID, credentials=credentials)
    return gcs_client.bucket(bucket_name)


def _download(cloud_file_path: str, local_file_path: Optional[str] = None) -> str:
    # if local_file_path is not specified saving in home dir
    if local_file_path is None:
        home_dir = os.getenv("HOME")
        local_file_path = os.path.join(home_dir, "Tensorleap_data3", BUCKET_NAME, cloud_file_path)
    # check if file already exists
    if os.path.exists(local_file_path):
        return local_file_path
    bucket = _connect_to_gcs_and_return_bucket(BUCKET_NAME)
    dir_path = os.path.dirname(local_file_path)
    os.makedirs(dir_path, exist_ok=True)
    blob = bucket.blob(cloud_file_path)
    blob.download_to_filename(local_file_path)
    return local_file_path


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


def subset_images():
    annFile = os.path.join(DIR, IMG_FOLDER, "train.json")
    fpath = _download(annFile)
    # initialize COCO api for instance annotations
    train = COCO(fpath)
    x_train_raw = load_set(coco=train, load_union=LOAD_UNION_CATEGORIES_IMAGES)

    # annFile = os.path.join(DIR, IMG_FOLDER, "val.json")
    # fpath = _download(annFile)
    # val = COCO(fpath)
    # x_val_raw = load_set(coco=val, load_union=LOAD_UNION_CATEGORIES_IMAGES)
    # 
    # annFile = os.path.join(DIR, IMG_FOLDER, "test.json")
    # fpath = _download(annFile)
    # test = COCO(fpath)
    # x_test_raw = load_set(coco=test, load_union=LOAD_UNION_CATEGORIES_IMAGES)

    annFile = os.path.join(DIR, IMG_FOLDER, "test.json")
    fpath = _download(annFile)
    val = COCO(fpath)
    x_val_raw = load_set(coco=val, load_union=LOAD_UNION_CATEGORIES_IMAGES)

    train_size = min(len(x_train_raw), TRAIN_SIZE)
    val_size = min(len(x_val_raw), VAL_SIZE)
    # test_size = min(len(x_test_raw), TEST_SIZE)
    np.random.seed(0)
    train_idx, val_idx = np.random.choice(len(x_train_raw), train_size), np.random.choice(len(x_val_raw),
                                                                                          val_size)  # , np.random.choice(len(x_test_raw), test_size)
    return [
        PreprocessResponse(length=train_size, data={'cocofile': train,
                                                    'samples': np.take(x_train_raw, train_idx),
                                                    'subdir': 'train'}),
        PreprocessResponse(length=val_size, data={'cocofile': val,
                                                  'samples': np.take(x_val_raw, val_idx),
                                                  'subdir': 'test'})  # ,
        # PreprocessResponse(length=val_size, data={'cocofile': test,
        #                                           'samples': np.take(x_test_raw, test_idx),
        #                                           'subdir': 'test'})
    ]


def unlabeled_preprocessing_func() -> PreprocessResponse:
    annFile = os.path.join(DIR, IMG_FOLDER, "val.json")
    fpath = _download(annFile)
    val = COCO(fpath)
    x_val_raw = load_set(coco=val, load_union=LOAD_UNION_CATEGORIES_IMAGES)
    val_size = min(len(x_val_raw), UL_SIZE)
    np.random.seed(0)
    val_idx = np.random.choice(len(x_val_raw), val_size)
    return PreprocessResponse(length=val_size, data={'cocofile': val,
                                                     'samples': np.take(x_val_raw, val_idx),
                                                     'subdir': 'val'})


def input_image(idx: int, data: PreprocessResponse) -> np.ndarray:
    """
    Returns a BGR image normalized and padded
    """
    data = data.data
    x = data['samples'][idx]
    filepath = f"{DIR}/{IMG_FOLDER}/images/{x['file_name']}"
    fpath = _download(filepath)
    # rescale
    # image = np.array(Image.open(fpath))
    image = np.array(Image.open(fpath).resize((IMAGE_SIZE[0], IMAGE_SIZE[1]), Image.BILINEAR)) / 255.
    # todo: add normalization
    return image


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
    res = cache_dicts['bbs'].get(str(idx) + data['subdir'])
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
    if len(cache_dicts['bbs'].keys()) > BATCH_SIZE:
        cache_dicts['bbs'] = {str(idx) + data['subdir']: bboxes}
    else:
        cache_dicts['bbs'][str(idx) + data['subdir']] = bboxes
    return bboxes


def polygon_to_bbox(vertices):
    xs = [x for i, x in enumerate(vertices) if i % 2 == 0]
    ys = [x for i, x in enumerate(vertices) if i % 2 != 0]

    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xs)
    max_y = max(ys)

    # Bounding box representation: (x, y, width, height)
    bbox = [(min_x + max_x) / 2., (min_y + max_y) / 2., max_x - min_x, max_y - min_y]

    return bbox


def compute_losses(obj_true: tf.Tensor, od_pred: tf.Tensor,
                   mask_true, instance_seg: tf.Tensor) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
    """
    Computes the sum of the classification (CE loss) and localization (regression) losses from all heads
    """
    res = cache_dicts['loss'].get(str(obj_true) + str(od_pred) + str(mask_true) + str(instance_seg))
    if res is not None:
        return res
    decoded = False if MODEL_FORMAT != "inference" else True
    class_list_reshaped, loc_list_reshaped = reshape_output_list(od_pred, decoded=decoded,
                                                                 image_size=IMAGE_SIZE)  # add batch
    # masks_pred = y_pred[..., CLASSES+5:]
    # od_pred = y_pred[..., :CLASSES+5]
    loss_l, loss_c, loss_o, loss_m = LOSS_FN(y_true=obj_true, y_pred=(loc_list_reshaped, class_list_reshaped),
                                             instance_seg=instance_seg, instance_true=mask_true)
    cache_dicts['loss'] = {
        str(obj_true) + str(od_pred) + str(mask_true) + str(instance_seg): (loss_l, loss_c, loss_o, loss_m)}
    return loss_l, loss_c, loss_o, loss_m


def instance_seg_loss(bb_gt: tf.Tensor, detection_pred: tf.Tensor,
                      mask_gt: tf.Tensor, segmentation_pred: tf.Tensor):  # return batch
    """
    Sums the classification and regression loss
    """
    loss_l, loss_c, loss_o, loss_m = compute_losses(bb_gt, detection_pred, mask_gt, segmentation_pred)
    combined_losses = [l + c + o + m for l, c, o, m in zip(loss_l, loss_c, loss_o, loss_m)]
    sum_loss = tf.reduce_sum(combined_losses, axis=0)
    return sum_loss


# -- decoders -- #


def bb_array_to_object(bb_array: Union[NDArray[float], tf.Tensor], iscornercoded: bool = True, bg_label: int = 0,
                       is_gt=False, masks: Optional[tf.Tensor] = None) -> List[BoundingBox]:
    """
    Assumes a (X,Y,W,H) Format for the BB text
    bb_array is (CLASSES,TOP_K,PROPERTIES) WHERE PROPERTIES =(conf,xmin,ymin,xmax,ymax)
    """
    bb_list = []
    mask_list = []
    if not isinstance(bb_array, np.ndarray):
        bb_array = bb_array.numpy()
    # fig, ax = plt.subplots(figsize=(6, 9)
    if len(bb_array.shape) == 3:
        bb_array = bb_array.reshape(-1, bb_array.shape[-1])
    if masks is not None and len(bb_array) > 0:
        if not isinstance(masks, np.ndarray):
            np_masks = masks.numpy()
        else:
            np_masks = masks
        if not is_gt:
            out_masks = (tf.sigmoid(np_masks @ bb_array[:, 6:].T).numpy() > 0.5).astype(float)
        else:
            out_masks = np.swapaxes(np.swapaxes(np_masks, 0, 1), 1, 2)
        no_object_masks = np.zeros_like(out_masks)
        h_factor = out_masks.shape[0]
        w_factor = out_masks.shape[1]
    for i in range(bb_array.shape[0]):
        if bb_array[i][-1] != bg_label:
            if iscornercoded:
                x, y, w, h = xyxy_to_xywh_format(bb_array[i][1:5])  # FIXED TOM
                # unormalize to image dimensions
            else:
                x, y = bb_array[i][0], bb_array[i][1]
                w, h = bb_array[i][2], bb_array[i][3]
            conf = 1 if is_gt else bb_array[i][0]
            curr_bb = BoundingBox(x=x, y=y, width=w, height=h, confidence=conf,
                                  label=CATEGORIES[int(bb_array[i][min(5, len(bb_array[i]) - 1)])])
            if masks is not None:
                if is_gt:
                    fixed_mask = out_masks[..., i]
                else:
                    no_object_masks[
                    int(max(y - h / 2, 0) * h_factor):int(min(y + h / 2, out_masks.shape[1]) * h_factor),
                    int(max(x - w / 2, 0) * h_factor):int(min(x + w / 2, out_masks.shape[1]) * w_factor), i] = 1
                    fixed_mask = out_masks[..., i] * no_object_masks[..., i]
                mask_list.append(fixed_mask.round().astype(int))
                # new_mask = Image.fromarray(fixed_mask).resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.NEAREST)
            # downscale box
            # crop mask according to prediction
            # increase mask to image size
            bb_list.append(curr_bb)
    return bb_list, mask_list


def bb_decoder(image, bb_prediction):
    """
    Overlays the BB predictions on the image
    """
    bb_object, _ = get_mask_list(bb_prediction, None, is_gt=False)
    return LeapImageWithBBox((image * 255).astype(np.float32), bb_object)


def gt_bb_decoder(image, bb_gt) -> LeapImageWithBBox:
    bb_object, _ = get_mask_list(bb_gt, None, is_gt=True)
    return LeapImageWithBBox((image * 255).astype(np.float32), bb_object)


def get_mask_list(data, masks, is_gt):
    res = cache_dicts['mask_list'].get(str(data) + str(masks) + str(is_gt))
    if res is not None:
        return res
    if is_gt:
        bb_object, mask_list = bb_array_to_object(data, iscornercoded=False, bg_label=BACKGROUND_LABEL,
                                                  is_gt=True,
                                                  masks=masks)
    else:
        from_logits = True if MODEL_FORMAT != "inference" else False
        decoded = False if MODEL_FORMAT != "inference" else True
        class_list_reshaped, loc_list_reshaped = reshape_output_list(
            np.reshape(data, (1, *data.shape)), decoded=decoded, image_size=IMAGE_SIZE)
        # add batch
        outputs = DECODER(loc_list_reshaped,
                          class_list_reshaped,
                          DEFAULT_BOXES,
                          from_logits=from_logits,
                          decoded=decoded,
                          )
        bb_object, mask_list = bb_array_to_object(outputs[0], iscornercoded=True, bg_label=BACKGROUND_LABEL,
                                                  masks=masks)
    if len(cache_dicts['mask_list'].keys()) > 4 * BATCH_SIZE:  # BATCH_SIZE*[FALSE/TRUE,MASK/NO-MASK]
        cache_dicts['mask_list'] = {str(data) + str(masks) + str(is_gt): (bb_object, mask_list)}
    else:
        cache_dicts['mask_list'][str(data) + str(masks) + str(is_gt)] = (bb_object, mask_list)
    return bb_object, mask_list


def get_mask_visualizer(image, bbs, masks):
    image_size = image.shape[:2]
    argmax_map = np.zeros(image_size, dtype=np.uint8)
    curr_idx = 1
    cats_dict = {}
    cats = []
    for bb, mask in zip(bbs, masks):
        if mask.shape != image_size:
            resize_mask = tf.image.resize(mask[..., None], image_size, tf.image.ResizeMethod.NEAREST_NEIGHBOR)[..., 0]
            if not isinstance(resize_mask, np.ndarray):
                resize_mask = resize_mask.numpy()
        else:
            resize_mask = mask
        resize_mask = resize_mask.astype(bool)
        label = bb.label
        argmax_map[resize_mask] = curr_idx
        instance_number = cats_dict.get(label, 0)
        cats_dict[label] = instance_number + 1
        cats += [f"{label}_{str(instance_number)}"]
        curr_idx += 1
    argmax_map[argmax_map == 0] = curr_idx
    argmax_map -= 1
    return LeapImageMask(mask=argmax_map.astype(np.uint8), image=image.astype(np.float32), labels=cats + ["background"])


def get_mask_visualizer_fixed_instances(image, bbs, masks):
    image_size = image.shape[:2]
    argmax_map = np.zeros(image_size, dtype=np.uint8)
    cats_dict = {}
    seperate_masks = []
    for bb, mask in zip(bbs, masks):
        if mask.shape != image_size:
            resize_mask = tf.image.resize(mask[..., None], image_size, tf.image.ResizeMethod.NEAREST_NEIGHBOR)[..., 0]
            if not isinstance(resize_mask, np.ndarray):
                resize_mask = resize_mask.numpy()
        else:
            resize_mask = mask
        resize_mask = resize_mask.astype(bool)
        label = bb.label
        instance_number = cats_dict.get(label, 0)
        # update counter if reach max instances we treat the last objects as one
        cats_dict[label] = instance_number + 1 if instance_number < MAX_INSTANCES_PER_CLASS else instance_number
        argmax_map[resize_mask] = CATEGORIES.index(label) * MAX_INSTANCES_PER_CLASS + cats_dict[label]  # curr_idx
        if bb.label == 'Object':
            seperate_masks.append(resize_mask)
    argmax_map[argmax_map == 0] = len(INSTANCES) + 1
    argmax_map -= 1
    return LeapImageMask(mask=argmax_map.astype(np.uint8), image=image.astype(np.float32),
                         labels=INSTANCES + ["background"]), seperate_masks


def mask_visualizer_gt(image, data, mask):
    bbs, masks = get_mask_list(data, mask, is_gt=True)
    # return get_mask_visualizer(image, bbs, masks)
    return get_mask_visualizer_fixed_instances(image, bbs, masks)[0]


def mask_visualizer_prediction(image, data, mask):
    bbs, masks = get_mask_list(data, mask, is_gt=False)
    # return get_mask_visualizer(image, bbs, masks)
    return get_mask_visualizer_fixed_instances(image, bbs, masks)[0]


def get_idx(index: int, subset: PreprocessResponse):
    return index


def classification_metric(bb_gt: tf.Tensor, detection_pred: tf.Tensor,
                          mask_gt: tf.Tensor, segmentation_pred: tf.Tensor):  # return batch
    _, loss_c, _, _ = compute_losses(bb_gt, detection_pred, mask_gt, segmentation_pred)
    return tf.reduce_sum(loss_c, axis=0)[:, 0]


def regression_metric(bb_gt: tf.Tensor, detection_pred: tf.Tensor,
                      mask_gt: tf.Tensor, segmentation_pred: tf.Tensor):  # return batch
    loss_l, _, _, _ = compute_losses(bb_gt, detection_pred, mask_gt, segmentation_pred)
    return tf.reduce_sum(loss_l, axis=0)[:, 0]  # shape of batch


def object_metric(bb_gt: tf.Tensor, detection_pred: tf.Tensor,
                  mask_gt: tf.Tensor, segmentation_pred: tf.Tensor):
    _, _, loss_o, _ = compute_losses(bb_gt, detection_pred, mask_gt, segmentation_pred)
    return tf.reduce_sum(loss_o, axis=0)[:, 0]  # shape of batch


def mask_metric(bb_gt: tf.Tensor, detection_pred: tf.Tensor,
                mask_gt: tf.Tensor, segmentation_pred: tf.Tensor):
    _, _, _, loss_m = compute_losses(bb_gt, detection_pred, mask_gt, segmentation_pred)
    return tf.reduce_sum(loss_m, axis=0)[:, 0]  # shape of batch


def multiple_mask_gt(image, data, mask):
    bbs, masks = get_mask_list(data, mask, is_gt=True)
    # return get_mask_visualizer(image, bbs, masks)
    return get_mask_visualizer_fixed_instances(image, bbs, masks)[1]


def multiple_mask_pred(image, data, mask):
    bbs, masks = get_mask_list(data, mask, is_gt=False)
    # return get_mask_visualizer(image, bbs, masks)
    return get_mask_visualizer_fixed_instances(image, bbs, masks)[1]


def ioa_mask(mask_containing, mask_contained):
    intersection_mask = mask_containing & mask_contained
    intersection = len(intersection_mask[intersection_mask])
    area = len(mask_contained[mask_contained])
    return intersection / max(area, 1)


# optim = tf.keras.optimizers.Adam()


def get_ioa_array(image, y_pred_bb, y_pred_mask, bb_gt, mask_gt, containing='pred'):
    hash_str = str(image) + str(y_pred_bb) + str(y_pred_mask) + str(bb_gt) + str(mask_gt) + str(containing)
    res = cache_dicts['get_ioa_array'].get(hash_str)
    if res is not None:
        return res
    prediction_masks = multiple_mask_pred(image, y_pred_bb, y_pred_mask)
    gt_masks = multiple_mask_gt(image, bb_gt, mask_gt)
    ioas = np.zeros((len(prediction_masks), len(gt_masks)))
    for i, pred_mask in enumerate(prediction_masks):
        for j, gt_mask in enumerate(gt_masks):
            if containing == 'pred':
                ioas[i, j] = ioa_mask(pred_mask, gt_mask)
            else:
                ioas[i, j] = ioa_mask(gt_mask, pred_mask)
    if len(cache_dicts['get_ioa_array'].keys()) > 2 * BATCH_SIZE:
        cache_dicts['get_ioa_array'] = {hash_str: ioas}
    else:
        cache_dicts['get_ioa_array'][hash_str] = ioas
    return ioas


def under_segmented(image, y_pred_bb, y_pred_mask, bb_gt, mask_gt):
    th = 0.8
    under_segmented_arr = []
    for i in range(image.shape[0]):
        ioas = get_ioa_array(image[i, ...], y_pred_bb[i, ...], y_pred_mask[i, ...], bb_gt[i, ...], mask_gt[i, ...],
                             containing='pred')
        matches_count = (ioas > th).astype(int).sum(axis=-1)
        is_under_segmented = float(len(matches_count[matches_count > 1]) > 0)
        under_segmented_arr.append(is_under_segmented)
    return tf.convert_to_tensor(under_segmented_arr)


def over_segmented(image, y_pred_bb, y_pred_mask, bb_gt, mask_gt):
    th = 0.8
    over_segmented_arr = []
    for i in range(image.shape[0]):
        ioas = get_ioa_array(image[i, ...], y_pred_bb[i, ...], y_pred_mask[i, ...], bb_gt[i, ...], mask_gt[i, ...],
                             containing='gt')
        matches_count = (ioas > th).astype(int).sum(axis=0)
        is_over_segmented = float(len(matches_count[matches_count > 1]) > 0)
        over_segmented_arr.append(is_over_segmented)
    return tf.convert_to_tensor(over_segmented_arr)


def get_idx(index: int, subset: PreprocessResponse) -> int:
    return index


def get_fname(index: int, subset: PreprocessResponse) -> str:
    data = subset.data
    x = data['samples'][index]
    return x['file_name']


def get_original_width(index: int, subset: PreprocessResponse) -> int:
    data = subset.data
    x = data['samples'][index]
    return x['width']


def get_original_height(index: int, subset: PreprocessResponse) -> int:
    data = subset.data
    x = data['samples'][index]
    return x['height']


def bbox_num(index: int, subset: PreprocessResponse) -> int:
    bbs = get_bbs(index, subset)
    number_of_bb = np.count_nonzero(bbs[..., -1] != BACKGROUND_LABEL)
    return number_of_bb


def avg_bb_area_metadata(index: int, subset: PreprocessResponse) -> float:
    bbs = get_bbs(index, subset)  # x,y,w,h
    valid_bbs = bbs[bbs[..., -1] != BACKGROUND_LABEL]
    areas = valid_bbs[:, 2] * valid_bbs[:, 3]
    return areas.mean()


def avg_bb_aspect_ratio(index: int, subset: PreprocessResponse) -> float:
    bbs = get_bbs(index, subset)
    valid_bbs = bbs[bbs[..., -1] != BACKGROUND_LABEL]
    aspect_ratios = valid_bbs[:, 2] / valid_bbs[:, 3]
    return aspect_ratios.mean()


def instances_num(index: int, subset: PreprocessResponse) -> float:
    bbs = get_bbs(index, subset)
    valid_bbs = bbs[bbs[..., -1] != BACKGROUND_LABEL]
    return float(valid_bbs.shape[0])


def object_instances_num(index: int, subset: PreprocessResponse) -> float:
    bbs = get_bbs(index, subset)
    label = CATEGORIES.index('Object')
    valid_bbs = bbs[bbs[..., -1] == label]
    return float(valid_bbs.shape[0])


def tote_instances_num(index: int, subset: PreprocessResponse) -> float:
    bbs = get_bbs(index, subset)
    label = CATEGORIES.index('Tote')
    valid_bbs = bbs[bbs[..., -1] == label]
    return float(valid_bbs.shape[0])


def avg_instance_percent(idx: int, data: PreprocessResponse) -> float:
    mask = get_masks(idx, data)
    bboxs = get_bbs(idx, data)
    valid_masks = mask[bboxs[..., -1] != BACKGROUND_LABEL]
    if valid_masks.size == 0:
        return 0.
    res = np.sum(valid_masks, axis=(1, 2))
    size = valid_masks[0, :, :].size
    return np.mean(np.divide(res, size))


def get_tote_instances_masks(idx: int, data: PreprocessResponse) -> Union[float, None]:
    mask = get_masks(idx, data)
    bboxs = get_bbs(idx, data)
    label = CATEGORIES.index('Tote')
    valid_masks = mask[bboxs[..., -1] == label]
    if valid_masks.size == 0:
        return None
    return valid_masks


def get_tote_instances_sizes(idx: int, data: PreprocessResponse) -> float:
    masks = get_tote_instances_masks(idx, data)
    if masks is None:
        return 0
    res = np.sum(masks, axis=(1, 2))
    size = masks[0, :, :].size
    return np.divide(res, size)


def tote_avg_instance_percent(idx: int, data: PreprocessResponse) -> float:
    sizes = get_tote_instances_sizes(idx, data)
    return float(np.mean(sizes))


def tote_std_instance_percent(idx: int, data: PreprocessResponse) -> float:
    sizes = get_tote_instances_sizes(idx, data)
    return float(np.std(sizes))


def get_cat_instances_seg_lst(idx: int, data: PreprocessResponse, cat: str) -> List[np.ma.array]:
    img = input_image(idx, data)
    if cat == "tote":
        masks = get_tote_instances_masks(idx, data)
    elif cat == "object":
        masks = get_object_instances_masks(idx, data)
    else:
        print('Error category not supported')
        return None
    if masks is None:
        return None
    if masks[0, ...].shape != IMAGE_SIZE:
        masks = tf.image.resize(masks[..., None], IMAGE_SIZE, tf.image.ResizeMethod.NEAREST_NEIGHBOR)[..., 0]
        masks = masks.numpy()
    instances = []
    for mask in masks:
        mask = np.broadcast_to(mask[..., np.newaxis], img.shape)
        masked_arr = np.ma.masked_array(img, mask)
        instances.append(masked_arr)
    return instances


def get_tote_instances_mean(idx: int, data: PreprocessResponse) -> float:
    instances = get_cat_instances_seg_lst(idx, data, "tote")
    if instances is None:
        return -1
    return np.array([i.mean() for i in instances]).mean()


def get_tote_instances_std(idx: int, data: PreprocessResponse) -> float:
    instances = get_cat_instances_seg_lst(idx, data, "tote")
    if instances is None:
        return -1
    return np.array([i.std() for i in instances]).std()


def get_object_instances_mean(idx: int, data: PreprocessResponse) -> float:
    instances = get_cat_instances_seg_lst(idx, data, "object")
    if instances is None:
        return -1
    return np.array([i.mean() for i in instances]).mean()


def get_object_instances_std(idx: int, data: PreprocessResponse) -> float:
    instances = get_cat_instances_seg_lst(idx, data, "object")
    if instances is None:
        return -1
    return np.array([i.std() for i in instances]).std()


def get_object_instances_masks(idx: int, data: PreprocessResponse) -> Union[np.ndarray, None]:
    mask = get_masks(idx, data)
    bboxs = get_bbs(idx, data)
    label = CATEGORIES.index('Object')
    valid_masks = mask[bboxs[..., -1] == label]
    if valid_masks.size == 0:
        return None
    return valid_masks


def get_object_instances_sizes(idx: int, data: PreprocessResponse) -> float:
    masks = get_object_instances_masks(idx, data)
    if masks is None:
        return 0
    res = np.sum(masks, axis=(1, 2))
    size = masks[0, :, :].size
    return np.divide(res, size)


def object_avg_instance_percent(idx: int, data: PreprocessResponse) -> float:
    sizes = get_object_instances_sizes(idx, data)
    return float(np.mean(sizes))


def object_std_instance_percent(idx: int, data: PreprocessResponse) -> float:
    sizes = get_object_instances_sizes(idx, data)
    return float(np.std(sizes))


def background_percent(idx: int, data: PreprocessResponse) -> float:
    masks = get_masks(idx, data)
    bboxs = get_bbs(idx, data)
    valid_masks = masks[bboxs[..., -1] != BACKGROUND_LABEL]
    if valid_masks.size == 0:
        return 1.
    res = np.sum(valid_masks, axis=0)
    size = valid_masks[0, :, :].size
    return float(np.round(np.divide(res[res == 0].size, size), 3))


def metric_small_bb_in_under_segment(image, y_pred_bb, y_pred_mask, bb_gt, mask_gt):  # bb_visualizer + gt_visualizer
    th = 0.8  # equivelan
    has_small_bbs = [0.] * image.shape[0]
    for i in range(image.shape[0]):
        ioas = get_ioa_array(image[i, ...], y_pred_bb[i, ...], y_pred_mask[i, ...], bb_gt[i, ...], mask_gt[i, ...],
                             containing='pred')
        th_arr = ioas > th
        matches_count = th_arr.astype(int).sum(axis=-1)
        relevant_bbs = np.argwhere(matches_count > 1)[..., 0]  # [Indices of bbs]
        relevant_gts = np.where(np.any((th_arr)[relevant_bbs], axis=0))[0]  # [Indices of gts]
        bb_gt_object, _ = get_mask_list(bb_gt, None, is_gt=True)
        new_gt_objects = remove_label_from_bbs(bb_gt_object, "Tote", "gt")
        new_bb_array = [new_gt_objects[i] for i in relevant_gts]
        for j in range(len(new_bb_array)):
            if new_bb_array[j].width * new_bb_array[j].height < SMALL_BBS_TH:
                has_small_bbs[i] = 1.
    return tf.convert_to_tensor(has_small_bbs)


def duplicate_bb(index: int, subset: PreprocessResponse):
    bbs_gt = get_bbs(index, subset)
    real_gt = bbs_gt[bbs_gt[..., 4] != BACKGROUND_LABEL]
    return int(real_gt.shape[0] != np.unique(real_gt, axis=0).shape[0])


def remove_label_from_bbs(bbs_object_array, removal_label, add_to_label):
    new_bb_arr = []
    for bb in bbs_object_array:
        if bb.label != removal_label:
            new_bb = deepcopy(bb)
            new_bb.label = new_bb.label + "_" + add_to_label
            new_bb_arr.append(new_bb)
    return new_bb_arr


def under_segmented_bb_visualizer(image, y_pred_bb, y_pred_mask, bb_gt, mask_gt):  # bb_visualizer + gt_visualizer
    th = 0.8
    rel_bbs = []
    ioas = get_ioa_array(image, y_pred_bb, y_pred_mask, bb_gt, mask_gt, containing='pred')
    th_arr = ioas > th
    matches_count = th_arr.astype(int).sum(axis=-1)
    relevant_bbs = np.argwhere(matches_count > 1)[..., 0]  # [Indices of bbs]
    relevant_gts = np.where(np.any((th_arr)[relevant_bbs], axis=0))[0]  # [Indices of gts]
    bb_pred_object, _ = get_mask_list(y_pred_bb, None, is_gt=False)
    new_bb_pred_object = remove_label_from_bbs(bb_pred_object, "Tote", "pred")
    bb_gt_object, _ = get_mask_list(bb_gt, None, is_gt=True)
    new_gt_objects = remove_label_from_bbs(bb_gt_object, "Tote", "gt")
    new_bb_array = [new_gt_objects[i] for i in relevant_gts] + [new_bb_pred_object[i] for i in relevant_bbs]
    return LeapImageWithBBox((image * 255).astype(np.float32), new_bb_array)


def over_segmented_bb_visualizer(image, y_pred_bb, y_pred_mask, bb_gt, mask_gt):  # bb_visualizer + gt_visualizer
    th = 0.8
    rel_bbs = []
    ioas = get_ioa_array(image, y_pred_bb, y_pred_mask, bb_gt, mask_gt, containing='gt')
    th_arr = ioas > th
    matches_count = th_arr.astype(int).sum(axis=0)
    relevant_gts = np.argwhere(matches_count > 1)[..., 0]  # [Indices of bbs]
    relevant_bbs = np.where(np.any(th_arr[..., relevant_gts], axis=1))[0]  # [Indices of gts]
    bb_pred_object, _ = get_mask_list(y_pred_bb, None, is_gt=False)
    new_bb_pred_object = remove_label_from_bbs(bb_pred_object, "Tote", "pred")
    bb_gt_object, _ = get_mask_list(bb_gt, None, is_gt=True)
    new_gt_objects = remove_label_from_bbs(bb_gt_object, "Tote", "gt")
    new_bb_array = [new_gt_objects[i] for i in relevant_gts] + [new_bb_pred_object[i] for i in relevant_bbs]
    return LeapImageWithBBox((image * 255).astype(np.float32), new_bb_array)


def calculate_overlap(box1, box2):
    # Extract coordinates of the bounding boxes
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the coordinates of the intersection rectangle
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = max(0, min(x1 + w1, x2 + w2) - x_intersection)
    h_intersection = max(0, min(y1 + h1, y2 + h2) - y_intersection)

    # Calculate the overlap area
    overlap_area = w_intersection * h_intersection

    return overlap_area


def count_small_bbs(idx: int, data: PreprocessResponse) -> float:
    bboxes = get_bbs(idx, data)
    obj_boxes = bboxes[bboxes[..., -1] == 0]
    areas = obj_boxes[..., 2] * obj_boxes[..., 3]
    return float(len(areas[areas < SMALL_BBS_TH]))


def calculate_iou_all_pairs(bboxes: np.ndarray, image_size: int) -> np.ndarray:
    # Reformat all bboxes to (x_min, y_min, x_max, y_max)
    bboxes = np.asarray([xywh_to_xyxy_format(bbox[:-1]) for bbox in bboxes]) * image_size
    num_bboxes = len(bboxes)
    # Calculate coordinates for all pairs
    x_min = np.maximum(bboxes[:, 0][:, np.newaxis], bboxes[:, 0])
    y_min = np.maximum(bboxes[:, 1][:, np.newaxis], bboxes[:, 1])
    x_max = np.minimum(bboxes[:, 2][:, np.newaxis], bboxes[:, 2])
    y_max = np.minimum(bboxes[:, 3][:, np.newaxis], bboxes[:, 3])

    # Calculate areas for all pairs
    area_a = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    area_b = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

    # Calculate intersection area for all pairs
    intersection_area = np.maximum(x_max - x_min, 0) * np.maximum(y_max - y_min, 0)

    # Calculate union area for all pairs
    union_area = area_a[:, np.newaxis] + area_b - intersection_area

    # Calculate IOU for all pairs
    iou = intersection_area / union_area
    iou = iou[np.triu_indices(num_bboxes, k=1)]
    return iou


def count_obj_bbox_occlusions(idx: int, data: PreprocessResponse, calc_avg_flag=False) -> float:
    img = input_image(idx, data)
    img_size = img.shape[0]
    occlusion_threshold = 0.2  # Example threshold value
    bboxes = get_bbs(idx, data)  # [x,y,w,h]
    label = CATEGORIES.index('Object')
    obj_bbox = bboxes[bboxes[..., -1] == label]
    if len(obj_bbox) == 0:
        return 0.0
    else:
        ious = calculate_iou_all_pairs(obj_bbox, img_size)
        occlusion_count = len(ious[ious > occlusion_threshold])
        if calc_avg_flag:
            return int(occlusion_count / len(obj_bbox))
        else:
            return occlusion_count


def avg_obj_bbox_occlusions(idx: int, data: PreprocessResponse) -> float:
    return count_obj_bbox_occlusions(idx, data, calc_avg_flag=True)


def count_obj_mask_occlusions(idx: int, data: PreprocessResponse) -> int:
    occlusion_threshold = 0.1  # Example threshold value

    masks = get_object_instances_masks(idx, data)
    if masks is None:
        return 0

    if masks[0, ...].shape != IMAGE_SIZE:
        masks = tf.image.resize(masks[..., None], IMAGE_SIZE, tf.image.ResizeMethod.NEAREST_NEIGHBOR)[..., 0]
        masks = masks.numpy()

    num_masks = len(masks)

    # Reshape masks to have a third dimension
    masks = np.expand_dims(masks, axis=-1)

    # Create tiled versions of masks for element-wise addition
    tiled_masks = np.broadcast_to(masks, (num_masks, num_masks, masks.shape[1], masks.shape[2], 1))
    tiled_masks_transposed = np.transpose(tiled_masks, axes=(1, 0, 2, 3, 4))

    # Compute overlay matrix
    overlay = tiled_masks + tiled_masks_transposed

    # Exclude same mask occlusions and duplicate pairs
    mask_indices = np.triu_indices(num_masks, k=1)
    overlay = overlay[mask_indices]

    intersection = np.sum(overlay > 1, axis=(-1, -2, -3))
    union = np.sum(overlay > 0, axis=(-1, -2, -3))

    iou = intersection / union

    # Count occlusions
    occlusion_count = int(np.sum(iou > occlusion_threshold))

    return occlusion_count


def non_binary_over_segmented(image, y_pred_bb, y_pred_mask, bb_gt, mask_gt):
    th = 0.8
    over_segmented_arr = []
    for i in range(image.shape[0]):
        ioas = get_ioa_array(image[i, ...], y_pred_bb[i, ...], y_pred_mask[i, ...], bb_gt[i, ...], mask_gt[i, ...],
                             containing='gt')
        matches_count = (ioas > th).astype(int).sum(axis=0)
        is_over_segmented = float(len(matches_count[matches_count > 1]))
        over_segmented_arr.append(is_over_segmented)
    return tf.convert_to_tensor(over_segmented_arr)


def non_binary_under_segmented(image, y_pred_bb, y_pred_mask, bb_gt, mask_gt):
    th = 0.8
    under_segmented_arr = []
    for i in range(image.shape[0]):
        ioas = get_ioa_array(image[i, ...], y_pred_bb[i, ...], y_pred_mask[i, ...], bb_gt[i, ...], mask_gt[i, ...],
                             containing='pred')
        matches_count = (ioas > th).astype(int).sum(axis=-1)
        is_under_segmented = float(len(matches_count[matches_count > 1]))
        under_segmented_arr.append(is_under_segmented)
    return tf.convert_to_tensor(under_segmented_arr)


def average_segments_num_over_segment(image, y_pred_bb, y_pred_mask, bb_gt, mask_gt):
    th = 0.8
    over_segmented_arr = []
    for i in range(image.shape[0]):
        ioas = get_ioa_array(image[i, ...], y_pred_bb[i, ...], y_pred_mask[i, ...], bb_gt[i, ...], mask_gt[i, ...],
                             containing='gt')
        matches_count = (ioas > th).astype(int).sum(axis=0)
        if len(matches_count[matches_count > 1]) > 0:
            is_over_segmented = float(matches_count[matches_count > 1].mean())
        else:
            is_over_segmented = 0.
        over_segmented_arr.append(is_over_segmented)
    return tf.convert_to_tensor(over_segmented_arr)


def average_segments_num_under_segmented(image, y_pred_bb, y_pred_mask, bb_gt, mask_gt):
    th = 0.8
    under_segmented_arr = []
    for i in range(image.shape[0]):
        ioas = get_ioa_array(image[i, ...], y_pred_bb[i, ...], y_pred_mask[i, ...], bb_gt[i, ...], mask_gt[i, ...],
                             containing='pred')
        matches_count = (ioas > th).astype(int).sum(axis=-1)
        if len(matches_count[matches_count > 1]) > 0:
            is_under_segmented = float(matches_count[matches_count > 1].mean())
        else:
            is_under_segmented = 0.
        under_segmented_arr.append(is_under_segmented)
    return tf.convert_to_tensor(under_segmented_arr)


def over_segment_avg_confidence(image, y_pred_bb, y_pred_mask, bb_gt, mask_gt):  # bb_visualizer + gt_visualizer
    th = 0.8
    conf_arr = []
    for i in range(image.shape[0]):
        ioas = get_ioa_array(image[i, ...], y_pred_bb[i, ...], y_pred_mask[i, ...], bb_gt[i, ...], mask_gt[i, ...],
                             containing='gt')
        th_arr = ioas > th
        matches_count = th_arr.astype(int).sum(axis=0)
        relevant_gts = np.argwhere(matches_count > 1)[..., 0]  # [Indices of bbs]
        relevant_bbs = np.where(np.any(th_arr[..., relevant_gts], axis=1))[0]  # [Indices of gts]
        bb_pred_object, _ = get_mask_list(y_pred_bb[i, ...], None, is_gt=False)
        new_bb_pred_object = remove_label_from_bbs(bb_pred_object, "Tote", "pred")
        bb_gt_object, _ = get_mask_list(bb_gt, None, is_gt=True)
        new_bb_array = [new_bb_pred_object[j] for j in relevant_bbs]
        if len(new_bb_array) > 0:
            avg_conf = np.array([new_bb_array[j].confidence for j in range(len(new_bb_array))]).mean()
        else:
            avg_conf = 0.
        conf_arr.append(avg_conf)
    return tf.convert_to_tensor(conf_arr)


# ---------------------------------------------------------binding------------------------------------------------------
# preprocess function
leap_binder.set_preprocess(subset_images)
# unlabeled data preprocess
leap_binder.set_unlabeled_data_preprocess(function=unlabeled_preprocessing_func)
# set input and gt
leap_binder.set_input(input_image, 'images')
leap_binder.set_ground_truth(get_bbs, 'bbs')
leap_binder.set_ground_truth(get_masks, 'masks')
# set prediction (object)
leap_binder.add_prediction('object detection',
                           ["x", "y", "w", "h", "obj"] +
                           [f"class_{i}" for i in range(CLASSES)] +
                           [f"mask_coeff_{i}" for i in range(32)])

# set prediction (segmentation)
leap_binder.add_prediction('segementation masks', [f"mask_{i}" for i in range(32)])

# set visualizers
leap_binder.set_visualizer(mask_visualizer_gt, 'gt_mask', LeapDataType.ImageMask)
leap_binder.set_visualizer(mask_visualizer_prediction, 'pred_mask', LeapDataType.ImageMask)
leap_binder.set_visualizer(gt_bb_decoder, 'bb_gt_decoder', LeapDataType.ImageWithBBox)
leap_binder.set_visualizer(bb_decoder, 'bb_decoder', LeapDataType.ImageWithBBox)
leap_binder.set_visualizer(under_segmented_bb_visualizer, 'under segment', LeapDataType.ImageWithBBox)
leap_binder.set_visualizer(over_segmented_bb_visualizer, 'over segment', LeapDataType.ImageWithBBox)

# set custom metrics
leap_binder.add_custom_metric(regression_metric, "Regression_metric")
leap_binder.add_custom_metric(classification_metric, "Classification_metric")
leap_binder.add_custom_metric(object_metric, "Objectness_metric")
leap_binder.add_custom_metric(mask_metric, "Mask metric")
leap_binder.add_custom_metric(over_segmented, "Over Segmented metric")
leap_binder.add_custom_metric(under_segmented, "Under Segmented metric")
leap_binder.add_custom_metric(metric_small_bb_in_under_segment, 'Small BB Under Segmtented metric')
leap_binder.add_custom_metric(non_binary_over_segmented, "Over Segmented Instances count")
leap_binder.add_custom_metric(non_binary_under_segmented, "Under Segmented Instances count")
leap_binder.add_custom_metric(average_segments_num_over_segment, "Average segments num Over Segmented")
leap_binder.add_custom_metric(average_segments_num_under_segmented, "Average segments num Under Segmented")
leap_binder.add_custom_metric(over_segment_avg_confidence, "Over Segment confidences")

# set metadata
leap_binder.set_metadata(get_idx, DatasetMetadataType.int, "idx_metadata")
leap_binder.set_metadata(get_fname, DatasetMetadataType.string, "fname_metadata")
leap_binder.set_metadata(get_original_width, DatasetMetadataType.int, "origin_width_metadata")
leap_binder.set_metadata(get_original_height, DatasetMetadataType.int, "origin_height_metadata")
leap_binder.set_metadata(instances_num, DatasetMetadataType.float, "instances_number_metadata")
leap_binder.set_metadata(tote_instances_num, DatasetMetadataType.float, "tote_number_metadata")
leap_binder.set_metadata(object_instances_num, DatasetMetadataType.float, "object_number_metadata")
leap_binder.set_metadata(avg_instance_percent, DatasetMetadataType.float, "avg_instance_size_metadata")
leap_binder.set_metadata(get_tote_instances_mean, DatasetMetadataType.float, "tote_instances_mean_metadata")
leap_binder.set_metadata(get_tote_instances_std, DatasetMetadataType.float, "tote_instances_std_metadata")
leap_binder.set_metadata(get_object_instances_mean, DatasetMetadataType.float, "object_instances_mean_metadata")
leap_binder.set_metadata(get_object_instances_std, DatasetMetadataType.float, "object_instances_std_metadata")
leap_binder.set_metadata(tote_avg_instance_percent, DatasetMetadataType.float, "tote_avg_instance_size_metadata")
leap_binder.set_metadata(tote_std_instance_percent, DatasetMetadataType.float, "tote_std_instance_size_metadata")
leap_binder.set_metadata(object_avg_instance_percent, DatasetMetadataType.float, "object_avg_instance_size_metadata")
leap_binder.set_metadata(object_std_instance_percent, DatasetMetadataType.float, "object_std_instance_size_metadata")
leap_binder.set_metadata(bbox_num, DatasetMetadataType.float, "bbox_number_metadata")
leap_binder.set_metadata(avg_bb_area_metadata, DatasetMetadataType.float, "bbox_area_metadata")
leap_binder.set_metadata(avg_bb_aspect_ratio, DatasetMetadataType.float, "bbox_aspect_ratio_metadata")
leap_binder.set_metadata(background_percent, DatasetMetadataType.float, "background_percent")
leap_binder.set_metadata(duplicate_bb, DatasetMetadataType.int, "duplicate_bb")
leap_binder.set_metadata(count_small_bbs, DatasetMetadataType.int, "small bbs number")
leap_binder.set_metadata(count_obj_bbox_occlusions, DatasetMetadataType.float, "count_total_obj_bbox_occlusions")
leap_binder.set_metadata(avg_obj_bbox_occlusions, DatasetMetadataType.int, "avg_obj_bbox_occlusions")
leap_binder.set_metadata(count_obj_mask_occlusions, DatasetMetadataType.float, "count_obj_mask_occlusions")
