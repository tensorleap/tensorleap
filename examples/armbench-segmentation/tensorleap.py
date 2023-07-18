import os
from typing import Union, List

import numpy as np
import tensorflow as tf
from PIL import Image

from code_loader import leap_binder
from code_loader.contract.enums import (
    DatasetMetadataType,
    LeapDataType
)

from code_loader.contract.datasetclasses import PreprocessResponse
from pycocotools.coco import COCO

from armbench_segmentation.config import CONFIG
from armbench_segmentation.preprocessing import load_set
from armbench_segmentation.utils.general_utils import count_obj_masks_occlusions, \
    count_obj_bbox_occlusions, extract_and_cache_bboxes
from armbench_segmentation.metrics import regression_metric, classification_metric, object_metric, \
    mask_metric, over_segmented, under_segmented, metric_small_bb_in_under_segment, non_binary_over_segmented, \
    non_binary_under_segmented, average_segments_num_over_segment, average_segments_num_under_segmented, \
    over_segment_avg_confidence
from armbench_segmentation.visualizers.visualizers import gt_bb_decoder, bb_decoder, \
    under_segmented_bb_visualizer, over_segmented_bb_visualizer
from armbench_segmentation.visualizers.visualizers_getters import mask_visualizer_gt, mask_visualizer_prediction


# ----------------------------------------------------data processing--------------------------------------------------
def subset_images() -> List[PreprocessResponse]:
    """
    This function returns the training and validation datasets in the format expected by tensorleap
    """
    ann_filepath = os.path.join(CONFIG['DIR'], CONFIG['IMG_FOLDER'], "train.json")
    # initialize COCO api for instance annotations
    train = COCO(ann_filepath)
    x_train_raw = load_set(coco=train, load_union=CONFIG['LOAD_UNION_CATEGORIES_IMAGES'])

    ann_filepath = os.path.join(CONFIG['DIR'], CONFIG['IMG_FOLDER'], "test.json")
    val = COCO(ann_filepath)
    x_val_raw = load_set(coco=val, load_union=CONFIG['LOAD_UNION_CATEGORIES_IMAGES'])

    train_size = min(len(x_train_raw), CONFIG['TRAIN_SIZE'])
    val_size = min(len(x_val_raw), CONFIG['VAL_SIZE'])
    np.random.seed(0)
    train_idx, val_idx = np.random.choice(len(x_train_raw), train_size), np.random.choice(len(x_val_raw), val_size)
    training_subset = PreprocessResponse(length=train_size, data={'cocofile': train,
                                                                  'samples': np.take(x_train_raw, train_idx),
                                                                  'subdir': 'train'})
    validation_subset = PreprocessResponse(length=val_size, data={'cocofile': val,
                                                                  'samples': np.take(x_val_raw, val_idx),
                                                                  'subdir': 'test'})
    return [training_subset, validation_subset]


def unlabeled_preprocessing_func() -> PreprocessResponse:
    """
    This function returns the unlabeled data split in the format expected by tensorleap
    """
    ann_filepath = os.path.join(CONFIG['DIR'], CONFIG['IMG_FOLDER'], "val.json")
    val = COCO(ann_filepath)
    x_val_raw = load_set(coco=val, load_union=CONFIG['LOAD_UNION_CATEGORIES_IMAGES'])
    val_size = min(len(x_val_raw), CONFIG['UL_SIZE'])
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
    filepath = f"{CONFIG['DIR']}/{CONFIG['IMG_FOLDER']}/images/{x['file_name']}"
    # rescale
    image = np.array(
        Image.open(filepath).resize((CONFIG['IMAGE_SIZE'][0], CONFIG['IMAGE_SIZE'][1]), Image.BILINEAR)) / 255.
    return image


def get_annotation_coco(idx: int, data: PreprocessResponse) -> np.ndarray:
    x = data['samples'][idx]
    coco = data['cocofile']
    ann_ids = coco.getAnnIds(imgIds=x['id'])
    anns = coco.loadAnns(ann_ids)
    return anns


def get_masks(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data
    MASK_SIZE = (160, 160)
    coco = data['cocofile']
    anns = get_annotation_coco(idx, data)
    masks = np.zeros([CONFIG['MAX_BB_PER_IMAGE'], *MASK_SIZE], dtype=np.uint8)
    for i in range(min(len(anns), CONFIG['MAX_BB_PER_IMAGE'])):
        ann = anns[i]
        mask = coco.annToMask(ann)
        mask = np.array(Image.fromarray(mask).resize((MASK_SIZE[0], MASK_SIZE[1]), Image.NEAREST))
        masks[i, ...] = mask
    return masks


def get_bbs(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data
    bboxes = extract_and_cache_bboxes(idx, data)
    return bboxes


# ----------------------------------------------------------metadata----------------------------------------------------
def get_cat_instances_seg_lst(idx: int, data: PreprocessResponse, cat: str) -> Union[List[np.ma.array], None]:
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
    if masks[0, ...].shape != CONFIG['IMAGE_SIZE']:
        masks = tf.image.resize(masks[..., None], CONFIG['IMAGE_SIZE'], tf.image.ResizeMethod.NEAREST_NEIGHBOR)[..., 0]
        masks = masks.numpy()
    instances = []
    for mask in masks:
        mask = np.broadcast_to(mask[..., np.newaxis], img.shape)
        masked_arr = np.ma.masked_array(img, mask)
        instances.append(masked_arr)
    return instances


def get_idx(index: int, subset: PreprocessResponse):
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
    number_of_bb = np.count_nonzero(bbs[..., -1] != CONFIG['BACKGROUND_LABEL'])
    return number_of_bb


def avg_bb_area_metadata(index: int, subset: PreprocessResponse) -> float:
    bbs = get_bbs(index, subset)  # x,y,w,h
    valid_bbs = bbs[bbs[..., -1] != CONFIG['BACKGROUND_LABEL']]
    areas = valid_bbs[:, 2] * valid_bbs[:, 3]
    return areas.mean()


def avg_bb_aspect_ratio(index: int, subset: PreprocessResponse) -> float:
    bbs = get_bbs(index, subset)
    valid_bbs = bbs[bbs[..., -1] != CONFIG['BACKGROUND_LABEL']]
    aspect_ratios = valid_bbs[:, 2] / valid_bbs[:, 3]
    return aspect_ratios.mean()


def instances_num(index: int, subset: PreprocessResponse) -> float:
    bbs = get_bbs(index, subset)
    valid_bbs = bbs[bbs[..., -1] != CONFIG['BACKGROUND_LABEL']]
    return float(valid_bbs.shape[0])


def object_instances_num(index: int, subset: PreprocessResponse) -> float:
    bbs = get_bbs(index, subset)
    label = CONFIG['CATEGORIES'].index('Object')
    valid_bbs = bbs[bbs[..., -1] == label]
    return float(valid_bbs.shape[0])


def tote_instances_num(index: int, subset: PreprocessResponse) -> float:
    bbs = get_bbs(index, subset)
    label = CONFIG['CATEGORIES'].index('Tote')
    valid_bbs = bbs[bbs[..., -1] == label]
    return float(valid_bbs.shape[0])


def avg_instance_percent(idx: int, data: PreprocessResponse) -> float:
    mask = get_masks(idx, data)
    bboxs = get_bbs(idx, data)
    valid_masks = mask[bboxs[..., -1] != CONFIG['BACKGROUND_LABEL']]
    if valid_masks.size == 0:
        return 0.
    res = np.sum(valid_masks, axis=(1, 2))
    size = valid_masks[0, :, :].size
    return np.mean(np.divide(res, size))


def get_tote_instances_masks(idx: int, data: PreprocessResponse) -> Union[float, None]:
    mask = get_masks(idx, data)
    bboxs = get_bbs(idx, data)
    label = CONFIG['CATEGORIES'].index('Tote')
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
    label = CONFIG['CATEGORIES'].index('Object')
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
    valid_masks = masks[bboxs[..., -1] != CONFIG['BACKGROUND_LABEL']]
    if valid_masks.size == 0:
        return 1.0
    res = np.sum(valid_masks, axis=0)
    size = valid_masks[0, :, :].size
    return float(np.round(np.divide(res[res == 0].size, size), 3))


def obj_bbox_occlusions_count(idx: int, data: PreprocessResponse, calc_avg_flag=False) -> float:
    occlusion_threshold = 0.2  # Example threshold value
    img = input_image(idx, data)
    bboxes = get_bbs(idx, data)  # [x,y,w,h]
    occlusions_count = count_obj_bbox_occlusions(img, bboxes, occlusion_threshold, calc_avg_flag)
    return occlusions_count


def obj_bbox_occlusions_avg(idx: int, data: PreprocessResponse) -> float:
    return obj_bbox_occlusions_count(idx, data, calc_avg_flag=True)


def obj_mask_occlusions_count(idx: int, data: PreprocessResponse) -> int:
    occlusion_threshold = 0.1  # Example threshold value
    masks = get_object_instances_masks(idx, data)
    occlusion_count = count_obj_masks_occlusions(masks, occlusion_threshold)
    return occlusion_count


def duplicate_bb(index: int, subset: PreprocessResponse):
    bbs_gt = get_bbs(index, subset)
    real_gt = bbs_gt[bbs_gt[..., 4] != CONFIG['BACKGROUND_LABEL']]
    return int(real_gt.shape[0] != np.unique(real_gt, axis=0).shape[0])


def count_small_bbs(idx: int, data: PreprocessResponse) -> float:
    bboxes = get_bbs(idx, data)
    obj_boxes = bboxes[bboxes[..., -1] == 0]
    areas = obj_boxes[..., 2] * obj_boxes[..., 3]
    return float(len(areas[areas < CONFIG['SMALL_BBS_TH']]))


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
                           [f"class_{i}" for i in range(CONFIG['CLASSES'])] +
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
leap_binder.set_metadata(obj_bbox_occlusions_count, DatasetMetadataType.float, "count_total_obj_bbox_occlusions")
leap_binder.set_metadata(obj_bbox_occlusions_avg, DatasetMetadataType.int, "avg_obj_bbox_occlusions")
leap_binder.set_metadata(obj_mask_occlusions_count, DatasetMetadataType.float, "count_obj_mask_occlusions")
