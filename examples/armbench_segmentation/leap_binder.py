
import os
from typing import Union, List, Dict

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
from armbench_segmentation.metrics import instance_seg_loss, compute_losses
from armbench_segmentation.visualizers.visualizers import gt_bb_decoder, bb_decoder, \
    under_segmented_bb_visualizer, over_segmented_bb_visualizer
from armbench_segmentation.visualizers.visualizers_getters import mask_visualizer_gt, mask_visualizer_prediction
from armbench_segmentation.utils.general_utils import get_mask_list, get_argmax_map_and_separate_masks
from armbench_segmentation.utils.ioa_utils import ioa_mask
from armbench_segmentation.metrics import over_under_segmented_metrics

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


def get_avg_bb_area(index: int, subset: PreprocessResponse) -> float:
    bbs = get_bbs(index, subset)  # x,y,w,h
    valid_bbs = bbs[bbs[..., -1] != CONFIG['BACKGROUND_LABEL']]
    areas = valid_bbs[:, 2] * valid_bbs[:, 3]
    return areas.mean()


def get_avg_bb_aspect_ratio(index: int, subset: PreprocessResponse) -> float:
    bbs = get_bbs(index, subset)
    valid_bbs = bbs[bbs[..., -1] != CONFIG['BACKGROUND_LABEL']]
    aspect_ratios = valid_bbs[:, 2] / valid_bbs[:, 3]
    return aspect_ratios.mean()


def get_instances_num(index: int, subset: PreprocessResponse) -> float:
    bbs = get_bbs(index, subset)
    valid_bbs = bbs[bbs[..., -1] != CONFIG['BACKGROUND_LABEL']]
    return float(valid_bbs.shape[0])


def get_object_instances_num(index: int, subset: PreprocessResponse) -> float:
    bbs = get_bbs(index, subset)
    label = CONFIG['CATEGORIES'].index('Object')
    valid_bbs = bbs[bbs[..., -1] == label]
    return float(valid_bbs.shape[0])


def get_tote_instances_num(index: int, subset: PreprocessResponse) -> float:
    bbs = get_bbs(index, subset)
    label = CONFIG['CATEGORIES'].index('Tote')
    valid_bbs = bbs[bbs[..., -1] == label]
    return float(valid_bbs.shape[0])


def get_avg_instance_percent(idx: int, data: PreprocessResponse) -> float:
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


def get_tote_avg_instance_percent(idx: int, data: PreprocessResponse) -> float:
    sizes = get_tote_instances_sizes(idx, data)
    return float(np.mean(sizes))


def get_tote_std_instance_percent(idx: int, data: PreprocessResponse) -> float:
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


def get_object_avg_instance_percent(idx: int, data: PreprocessResponse) -> float:
    sizes = get_object_instances_sizes(idx, data)
    return float(np.mean(sizes))


def get_object_std_instance_percent(idx: int, data: PreprocessResponse) -> float:
    sizes = get_object_instances_sizes(idx, data)
    return float(np.std(sizes))


def get_background_percent(idx: int, data: PreprocessResponse) -> float:
    masks = get_masks(idx, data)
    bboxs = get_bbs(idx, data)
    valid_masks = masks[bboxs[..., -1] != CONFIG['BACKGROUND_LABEL']]
    if valid_masks.size == 0:
        return 1.0
    res = np.sum(valid_masks, axis=0)
    size = valid_masks[0, :, :].size
    return float(np.round(np.divide(res[res == 0].size, size), 3))


def get_obj_bbox_occlusions_count(idx: int, data: PreprocessResponse, calc_avg_flag=False) -> float:
    occlusion_threshold = 0.2  # Example threshold value
    img = input_image(idx, data)
    bboxes = get_bbs(idx, data)  # [x,y,w,h]
    occlusions_count = count_obj_bbox_occlusions(img, bboxes, occlusion_threshold, calc_avg_flag)
    return occlusions_count


def get_obj_bbox_occlusions_avg(idx: int, data: PreprocessResponse) -> float:
    return get_obj_bbox_occlusions_count(idx, data, calc_avg_flag=True)


def get_obj_mask_occlusions_count(idx: int, data: PreprocessResponse) -> int:
    occlusion_threshold = 0.1  # Example threshold value
    masks = get_object_instances_masks(idx, data)
    occlusion_count = count_obj_masks_occlusions(masks, occlusion_threshold)
    return occlusion_count


def count_duplicate_bbs(index: int, subset: PreprocessResponse):
    bbs_gt = get_bbs(index, subset)
    real_gt = bbs_gt[bbs_gt[..., 4] != CONFIG['BACKGROUND_LABEL']]
    return int(real_gt.shape[0] != np.unique(real_gt, axis=0).shape[0])


def count_small_bbs(idx: int, data: PreprocessResponse) -> float:
    bboxes = get_bbs(idx, data)
    obj_boxes = bboxes[bboxes[..., -1] == 0]
    areas = obj_boxes[..., 2] * obj_boxes[..., 3]
    return float(len(areas[areas < CONFIG['SMALL_BBS_TH']]))


def metadata_dict(idx: int, data: PreprocessResponse) -> Dict[str, Union[float, int, str]]:

    metadata_functions = {
        "idx": get_idx,
        "fname": get_fname,
        "origin_width": get_original_width,
        "origin_height": get_original_height,
        "instances_number": get_instances_num,
        "tote_number": get_tote_instances_num,
        "object_number": get_object_instances_num,
        "avg_instance_size": get_avg_instance_percent,
        "tote_instances_mean": get_tote_instances_mean,
        "tote_instances_std": get_tote_instances_std,
        "object_instances_mean": get_object_instances_mean,
        "object_instances_std": get_object_instances_std,
        "tote_avg_instance_size": get_tote_avg_instance_percent,
        "tote_std_instance_size": get_tote_std_instance_percent,
        "object_avg_instance_size": get_object_avg_instance_percent,
        "object_std_instance_size": get_object_std_instance_percent,
        "bbox_number": bbox_num,
        "bbox_area": get_avg_bb_area,
        "bbox_aspect_ratio": get_avg_bb_aspect_ratio,
        "background_percent": get_background_percent,
        "duplicate_bb": count_duplicate_bbs,
        "small_bbs_number": count_small_bbs,
        "count_total_obj_bbox_occlusions": get_obj_bbox_occlusions_count,
        "avg_obj_bbox_occlusions": get_obj_bbox_occlusions_avg,
        "count_obj_mask_occlusions": get_obj_mask_occlusions_count
    }
    res = dict()
    for func_name, func in metadata_functions.items():
        res[func_name] = func(idx, data)
    return res


def general_metrics_dict(bb_gt: tf.Tensor, detection_pred: tf.Tensor,
                         mask_gt: tf.Tensor, segmentation_pred: tf.Tensor) -> Dict[str, tf.Tensor]:
    reg_met, class_met, obj_met, mask_met = compute_losses(bb_gt, detection_pred, mask_gt, segmentation_pred)
    res = {
        "Regression_metric": tf.reduce_sum(reg_met, axis=0)[:, 0],
        "Classification_metric": tf.reduce_sum(class_met, axis=0)[:, 0],
        "Objectness_metric": tf.reduce_sum(obj_met, axis=0)[:, 0],
        "Mask_metric": tf.reduce_sum(mask_met, axis=0)[:, 0],
    }
    return res


def segmentation_metrics_dict(image: tf.Tensor, y_pred_bb: tf.Tensor, y_pred_mask: tf.Tensor, bb_gt: tf.Tensor,
                              mask_gt: tf.Tensor) -> Dict[str, Union[int, float]]:
    bs = bb_gt.shape[0]
    bb_mask_gt = [get_mask_list(bb_gt[i, ...], mask_gt[i, ...], is_gt=True) for i in range(bs)]
    bb_mask_pred = [get_mask_list(y_pred_bb[i, ...], y_pred_mask[i, ...], is_gt=False) for i in range(bs)]
    sep_mask_pred = [get_argmax_map_and_separate_masks(image[i, ...], bb_mask_pred[i][0],
                                       bb_mask_pred[i][1])['separate_masks'] for i in range(bs)]
    sep_mask_gt = [get_argmax_map_and_separate_masks(image[i, ...], bb_mask_gt[i][0],
                                       bb_mask_gt[i][1])['separate_masks'] for i in range(bs)]
    pred_gt_ioas = [np.array([[ioa_mask(pred_mask, gt_mask) for gt_mask in sep_mask_gt[i]]
                              for pred_mask in sep_mask_pred[i]]) for i in range(bs)]
    gt_pred_ioas = [np.array([[ioa_mask(gt_mask, pred_mask) for gt_mask in sep_mask_gt[i]]
                             for pred_mask in sep_mask_pred[i]]) for i in range(bs)]
    gt_pred_ioas_t =[arr.transpose() for arr in gt_pred_ioas]
    over_seg_bool, over_seg_count, avg_segments_over, _, over_conf =\
        over_under_segmented_metrics(gt_pred_ioas_t, get_avg_confidence=True, bb_mask_object_list=bb_mask_pred)
    under_seg_bool, under_seg_count, avg_segments_under, under_small_bb, _ =\
        over_under_segmented_metrics(pred_gt_ioas, count_small_bbs=True, bb_mask_object_list=bb_mask_gt)
    res = {
        "Over_Segmented_metric": over_seg_bool,
        "Under_Segmented_metric": under_seg_bool,
        "Small_BB_Under_Segmtented": under_small_bb,
        "Over_Segmented_Instances_count": over_seg_count,
        "Under_Segmented_Instances_count": under_seg_count,
        "Average_segments_num_Over_Segmented": avg_segments_over,
        "Average_segments_num_Under_Segmented": avg_segments_under,
        "Over_Segment_confidences": over_conf
    }
    return res


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
# set custom loss
leap_binder.add_custom_loss(instance_seg_loss, 'instance_seg loss')

# set visualizers
leap_binder.set_visualizer(mask_visualizer_gt, 'gt_mask', LeapDataType.ImageMask)
leap_binder.set_visualizer(mask_visualizer_prediction, 'pred_mask', LeapDataType.ImageMask)
leap_binder.set_visualizer(gt_bb_decoder, 'bb_gt_decoder', LeapDataType.ImageWithBBox)
leap_binder.set_visualizer(bb_decoder, 'bb_decoder', LeapDataType.ImageWithBBox)
leap_binder.set_visualizer(under_segmented_bb_visualizer, 'under segment', LeapDataType.ImageWithBBox)
leap_binder.set_visualizer(over_segmented_bb_visualizer, 'over segment', LeapDataType.ImageWithBBox)

# set custom metrics
leap_binder.add_custom_metric(general_metrics_dict, 'general_metrics')
leap_binder.add_custom_metric(segmentation_metrics_dict, 'segmentation_metrics')

# set metadata
leap_binder.set_metadata(metadata_dict, name='metadata')
