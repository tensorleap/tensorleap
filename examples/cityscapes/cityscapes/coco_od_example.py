import os
from typing import List, Optional, Callable, Union, Tuple, Dict
from functools import lru_cache
import json
from io import BytesIO
import math
import tensorflow as tf
import numpy as np
from numpy.typing import NDArray
import cv2
from PIL import Image
from google.cloud import storage
from google.cloud.storage import Bucket
from google.auth.credentials import AnonymousCredentials
from pycocotools.coco import COCO
from matplotlib.figure import Figure
from matplotlib import patches
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.visualizer_classes import LeapImage
from code_loader.contract.enums import (
    DatasetMetadataType,
    LeapDataType
)
from code_loader.contract.visualizer_classes import LeapImageWithBBox
from code_loader.contract.responsedataclasses import BoundingBox


# --------------------------------------Decoder functions------------------------------- #


def xyxy_to_xywh_format(boxes: Union[NDArray[float], tf.Tensor]) -> Union[NDArray[float], tf.Tensor]:
    """
    This gets bb in a [X,Y,W,H] format and transforms them into an [Xmin, Ymin, Xmax, Ymax] format
    :param boxes: [Num_boxes, 4] of type ndarray or tensor
    :return:
    """
    min_xy = (boxes[..., :2] + boxes[..., 2:]) / 2
    max_xy = (boxes[..., 2:] - boxes[..., :2])
    if isinstance(boxes, tf.Tensor):
        result = tf.concat([min_xy, max_xy], -1)
    else:
        result = np.concatenate([min_xy, max_xy], -1)
    return result


def xywh_to_xyxy_format(boxes: Union[NDArray[float], tf.Tensor]) -> Union[NDArray[float], tf.Tensor]:
    """
    This gets bb in a [X,Y,W,H] format and transforms them into an [Xmin, Ymin, Xmax, Ymax] format
    :param boxes: [Num_boxes, 4] of type ndarray or tensor
    :return:
    """
    min_xy = boxes[..., :2] - boxes[..., 2:] / 2
    max_xy = boxes[..., :2] + boxes[..., 2:] / 2
    if isinstance(boxes, tf.Tensor):
        result = tf.concat([min_xy, max_xy], -1)
    else:
        result = np.concatenate([min_xy, max_xy], -1)
    return result


def decode_bboxes(loc_pred: tf.Tensor, priors: tf.Tensor, variances=1):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc_pred (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors, 4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions as follows:
        x_pred = x_anchor + x_anchor*x_delta_pred
        y_pred = y_anchor + y_anchor*y_delta_pred
        w_pred = exp(w_delta_pred*var)*w_anchor
        h_pred = exp(h_delta_pred*var)*h_anchor
    """
    MAX_CLIP = 4.135166556742356
    log_preds = tf.clip_by_value(loc_pred[:, 2:], clip_value_max=MAX_CLIP, clip_value_min=-np.inf)
    boxes = tf.concat([
        priors[:, :2] + loc_pred[:, :2] * variances * priors[:, 2:],
        priors[:, 2:] * tf.math.exp(log_preds * variances)
    ], 1)
    return xywh_to_xyxy_format(boxes)


class DecoderDetectron:
    """At test time, Detect is the final layer of SSD.
    Consists of 4 steps:
    Bounding Boxes Decoding,
    Confidence Thresholding,
    Non-Max Suppression,
    Top-K Filtering.
    """

    def __init__(self, num_classes: int, background_label: int, top_k: int, conf_thresh: float, nms_thresh: float):
        self.num_classes = num_classes
        self.background_label = background_label
        self.top_k = top_k
        # Parameters used in nms.
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.variance = 1

    def __call__(self, loc_data, conf_data, prior_data, from_logits=True):
        """
        Args:
            loc_data: (tensor) Location preds from loc layers
                Shape: [batch, num_priors*4]
            conf_data: (tensor) Shape: Confidence preds from confidence layers
                Shape: [batch*num_priors, num_classes]
            prior_data: (tensor) Prior boxes and variances? from priorbox layers
                Shape: [1, num_priors, 4]
        """
        # loc_data: (batch_size, num_priors, 4)
        # conf_data: (batch_size, num_priors, num_classes)
        MAX_RETURNS = 100
        MAX_CANDIDATES_PER_LAYER = 10
        classes_num = conf_data[0].shape[-1]
        conf_preds = [tf.transpose(a=layer_conf, perm=[0, 2, 1]) for layer_conf in
                      conf_data]  # (batch_size, num_classes, num_priors)
        outputs = []
        for i in range(tf.shape(loc_data[0])[0]):
            loc = [loc_e[i, ...] for loc_e in loc_data]
            conf = [conf_e[i, ...] for conf_e in conf_preds]
            if from_logits:
                conf = [tf.math.sigmoid(layer_conf) for layer_conf in conf]
            class_selections = [[] for i in range(classes_num)]
            for l_loc, l_conf, l_prior in zip(loc, conf, prior_data):
                classes = tf.argmax(l_conf, axis=0)
                max_scores = tf.reduce_max(l_conf, axis=0)
                mask = max_scores > self.conf_thresh
                non_zero_indices = tf.where(mask)[:, 0]
                scores_masked = max_scores[mask]
                if len(scores_masked) > MAX_CANDIDATES_PER_LAYER:
                    best_scores, best_indices = tf.math.top_k(scores_masked, k=MAX_CANDIDATES_PER_LAYER)
                else:
                    best_scores = scores_masked
                    best_indices = np.arange(len(scores_masked))
                topk_indices = tf.gather(non_zero_indices, best_indices)
                selected_loc = l_loc[topk_indices, :]
                selected_scores = best_scores
                selected_prior = l_prior[topk_indices, :]
                selected_decoded = decode_bboxes(selected_loc, selected_prior,
                                                 self.variance)  # (num_priors, 4)  (xmin, ymin, xmax, ymax) - THIS WORKS
                selected_classes = tf.gather(classes, topk_indices)
                for k in range(len(selected_classes)):
                    class_selections[selected_classes[k]].append(
                        (selected_scores[k], *selected_decoded[k, :], selected_classes[k]))
            final_preds = []
            for i in range(classes_num):
                if len(class_selections[i]) > 0:
                    np_selection = np.array(class_selections[i])
                    boxes = np_selection[:, 1:5]
                    scores = np_selection[:, 0]
                    selected_indices = tf.image.non_max_suppression(boxes=boxes,
                                                                    scores=scores,
                                                                    max_output_size=self.top_k,
                                                                    iou_threshold=self.nms_thresh)
                    final_preds.append(np_selection[selected_indices, :])
            predictions = np.concatenate(final_preds, axis=0)
            predictions = predictions[:MAX_RETURNS, ...]
            outputs.append(predictions)
        return outputs


# --------------------------------------End decoder function---------------------------- #
# -------------------------------------- Visualizers helper ---------------------------- #
def matplotlib_to_numpy(fig):
    """
    This gets a matplotlib figure and returns a numpy array that captures the information in the figure
    :param fig:
    :return:
    """
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# ------------------------------  End visualizer helper ------------------------------- #
# ---------------------------------------- tf function ---------------------------------#
def clip_by_value(t, clip_value_min=None, clip_value_max=None):
    t = tf.convert_to_tensor(t)
    if clip_value_min is None and clip_value_max is None:
        raise ValueError()
    if clip_value_max is None:
        return tf.math.maximum(t, clip_value_min)
    if clip_value_min is None:
        return tf.math.minimum(t, clip_value_max)
    t_max = tf.math.maximum(t, clip_value_min)
    return tf.math.minimum(t_max, clip_value_max)


def log_sum_exp(x):
    x_max = tf.math.reduce_max(x)
    return tf.math.log(tf.math.reduce_sum(input_tensor=tf.math.exp(x - x_max), axis=1, keepdims=True)) + x_max


# ------------------------------------------tf function ---------------------------------#
# ---------------------------------------- IOU -----------------------------------------#
def intersect(box_a: tf.Tensor, box_b: tf.Tensor):
    """

    :param box_a: Tensor, shape: (A, 4)
    :param box_b: Tensor, shape: (B, 4)
    :return: intersetction of box_a and box_b shape: (A, B)
    """
    min_xy = tf.math.minimum(tf.expand_dims(box_a[:, 2:], axis=1),
                             tf.expand_dims(box_b[:, 2:], axis=0))  # (right_bottom)
    max_xy = tf.math.maximum(tf.expand_dims(box_a[:, :2], axis=1),
                             tf.expand_dims(box_b[:, :2], axis=0))  # (left_top)

    inter = clip_by_value(t=(min_xy - max_xy), clip_value_min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a: tf.Tensor, box_b: tf.Tensor):
    """
    computes the IOU scores between two set of bb (box_a, box_b)
    :param box_a: Tensor, GT bounding boxes, shape: (a, 4)
    :param box_b: Tensor, other bounding boxes, shape: (b, 4)
    :return: Tensor, shape: (a, b)
    """
    inter = intersect(box_a, box_b)
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_a = tf.expand_dims(area_a, axis=1)
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
    area_b = tf.expand_dims(area_b, axis=0)
    union = area_a + area_b - inter
    ratio = inter / union
    non_negativ_ratio = tf.where(inter > 0, x=ratio, y=0)
    return non_negativ_ratio


def encode_bboxes(matched: tf.Tensor, priors: tf.Tensor, variances: Tuple[int, int]) -> tf.Tensor:
    """
    This encodes a [X,Y,W,H] GT into a list of matches priors s.t.

    :param matched: Tensor - matched GT for each prior [N_PRIORS,4]
    :param priors: Tensor - Priors [N_priors,4]
    :param variances: Variances used
    :return:
        encoded bounding box gt Tensor
    """
    g_cxcy = (matched[:, :2] - priors[:, :2]) / (variances[0] * priors[:, 2:])
    g_wh = tf.math.log((matched[:, 2:]) / priors[:, 2:]) / variances[1]
    return tf.concat([g_cxcy, g_wh], 1)


def match(threshold: float, truths: tf.Tensor, priors: tf.Tensor,
          variances: Tuple[int, int], labels: tf.Tensor, background_label) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Matches between the GT and the anchors
    :param threshold:
    :param truths: (N_truths,4) - (X,Y,W,H)
    :param priors: (N_priors,4) - (X,Y,W,H)
    :param variances:
    :param labels: (N_truths) GT class
    :param background_label: int - the background label
    :return: loc (Tensor) [Npriors, 4], pred_label (Tensor) [Npriors]
    """
    # compute jaccard and best prior overlap and truth overlap
    overlaps = jaccard(xywh_to_xyxy_format(truths), xywh_to_xyxy_format(priors))  # (N_TRUTHS, N_PRIORS)
    best_prior_idx = tf.math.argmax(overlaps, axis=1)  # (NTRUTHS,)
    best_truth_overlap = tf.math.reduce_max(overlaps, axis=0, keepdims=True)  # (1, N_PRIORS)
    best_truth_idx = tf.math.argmax(overlaps, axis=0)  # (N_PRIORS,)
    # rates priors by GT overlap
    tf.squeeze(best_truth_overlap)
    tf.expand_dims(best_prior_idx, axis=1)
    tf.fill(dims=[tf.shape(best_prior_idx)[0]], value=2.0)
    best_truth_overlap = tf.tensor_scatter_nd_update(tensor=tf.squeeze(best_truth_overlap),
                                                     indices=tf.expand_dims(best_prior_idx, axis=1),
                                                     updates=tf.fill(dims=[tf.shape(best_prior_idx)[0]],
                                                                     value=2.0))  # (N_PRIORS)
    best_truth_overlap = tf.expand_dims(best_truth_overlap, axis=0)
    tf.expand_dims(best_prior_idx, axis=1)
    tf.range(start=0, limit=tf.shape(best_prior_idx)[0], delta=1,
             dtype=tf.int64)
    # For every PRIOR what is the best GT IDX
    best_truth_idx = tf.tensor_scatter_nd_update(tensor=best_truth_idx,
                                                 indices=tf.expand_dims(best_prior_idx, axis=1),
                                                 updates=tf.range(start=0, limit=tf.shape(best_prior_idx)[0], delta=1,
                                                                  dtype=tf.int64))
    # FOR EACH GT, replace the value of the best fitting prior with the GT INDEX
    # THIS RATES ALL GT ACCORDING TO WHICH RESULT IN HIGHEST JACACRD
    matches = tf.gather(params=truths, indices=best_truth_idx)  # GT for each PRIOR (N_PRIOR, 4)
    pred_label = tf.gather(params=labels, indices=best_truth_idx)  # THIS IS THE BEST LABELS
    pred_label = tf.where(condition=best_truth_overlap < threshold, x=background_label, y=tf.cast(pred_label,
                                                                                                  tf.int32))  # eliminates low threshold
    pred_label = tf.squeeze(pred_label)  # (Nprior)
    loc = encode_bboxes(matches, priors, variances)
    return loc, pred_label


# -----------------------------------------IOU end -------------------------------------#
# -------------------------------------------- MultiBox loss ---------------------------#
def true_coords_labels(idx: int, y_true: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    y_true shape  Tensor, shape: (batch_size, MAX_BOXES_PER_IMAGE, 5) (5 last channels are (xmin, ymin, xmax, ymax, class_index))
    This removes the 0 class from the ground_truth_labels
    :param idx:
    :param y_true:
    :return:
    """
    y_true = y_true[idx]
    mask = y_true[:, -1] != BACKGROUND_LABEL  # class index should be greater than 0
    masked_true = tf.boolean_mask(y_true, mask)
    true_coords = masked_true[:, :-1]
    true_labels = masked_true[:, -1]
    return true_coords, true_labels


class MultiBoxLossDetectron:
    """
    num classes - the number of classes to detect
    default_boxes - the anchors at all heads
    overlap_thresh - the threshold of IOU overwhich a match is positive
    neg_pos - the ratio of negative:positive samples
    background_label - should be the last idx
    loss - a string of which loss to use: 'focal' or 'ce'
    """

    def __init__(self, num_classes, default_boxes, overlap_thresh, neg_pos, background_label, loss='focal'):
        self.background_label = background_label
        self.default_boxes = [tf.convert_to_tensor(box_arr) for box_arr in default_boxes]
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.variance = (1, 1)
        self.negpos_ratio = neg_pos
        self.loss = loss

    def _focal_loss(self, one_hot_labels, predictions, gamma=2., alpha=0.25):
        ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(one_hot_labels, predictions)  # this penelize bg
        p = tf.sigmoid(predictions)
        p_t = p * one_hot_labels + (1 - p) * (1 - one_hot_labels)
        loss = ce_loss * ((1 - p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * one_hot_labels + (1 - alpha) * (1 - one_hot_labels)
            loss = alpha_t * loss
        return tf.reduce_sum(loss)

    def __call__(self, y_true: tf.Tensor, y_pred: Tuple[List[tf.Tensor], List[tf.Tensor]]) -> \
            Tuple[List[tf.Tensor], List[tf.Tensor]]:

        """
        Computes
        :param y_true:  Tensor, shape: (batch_size, MAX_BOXES_PER_IMAGE, 5(x, y, w, h, class_index))
        :param y_pred:  Tuple, (loc, conf) loc:
        :return: l_loss a list of NUM_FEATURES, each item a tensor with BATCH_SIZE legth
        :return: c_loss a list of NUM_FEATURES, each item a tensor with BATCH_SIZE legth
        """
        loc_data, conf_data = y_pred
        num = y_true.shape[0]
        l_losses = []
        c_losses = []
        smooth_l1_loss_fn = tf.keras.losses.Huber(delta=1.0, reduction=tf.keras.losses.Reduction.SUM)
        for i in range(len(self.default_boxes)):
            default_box_layer = self.default_boxes[i]
            loc_data_layer = loc_data[i]
            conf_data_layer = conf_data[i]
            priors = default_box_layer[:loc_data_layer.shape[1], :]
            # GT boxes
            loc_t = []
            conf_t = []
            priors = tf.cast(priors, dtype=tf.float32)  #
            y_true = tf.cast(y_true, dtype=tf.float32)
            for idx in range(num):
                truths, labels = true_coords_labels(idx, y_true)  #
                loc, conf = match(threshold=self.threshold, truths=truths, priors=priors, variances=self.variance,
                                  labels=labels, background_label=self.background_label)
                loc_t.append(loc)
                conf_t.append(conf)
            loc_t = tf.stack(values=loc_t, axis=0)  # this is the location predictions (relative and logged)
            conf_t = tf.stack(values=conf_t, axis=0)  # these are the labels

            pos = conf_t != self.background_label
            # loss: Smooth L1 loss
            pos_idx = tf.expand_dims(pos, axis=-1)
            pos_idx = tf.broadcast_to(pos_idx, shape=loc_data_layer.shape)

            ## apply per sample
            loss_l_list = []
            for j in range(num):
                loc_p = tf.boolean_mask(tensor=loc_data_layer[j, ...], mask=pos_idx[j, ...])
                loc_t_single = tf.boolean_mask(tensor=loc_t[j, ...], mask=pos_idx[j, ...])
                loc_p = tf.reshape(loc_p, shape=(-1, 4))
                loc_t_single = tf.reshape(loc_t_single, shape=(-1, 4))
                loss_l = smooth_l1_loss_fn(y_true=loc_t_single, y_pred=loc_p)
                loss_l_list.append(loss_l)
            loss_l_tensor = tf.stack(values=loss_l_list, axis=0)  # t
            batch_conf = tf.reshape(conf_data_layer, shape=(-1, self.num_classes))  # (69856, 21)
            conf_t = tf.cast(conf_t, dtype=tf.int32)  # (8, 8732)
            loss_c = log_sum_exp(batch_conf) - tf.gather(params=batch_conf, indices=tf.reshape(conf_t, shape=(-1, 1)),
                                                         batch_dims=1)  # This should be cross entropy

            # Hard Negative Mining
            loss_c = tf.reshape(loss_c, shape=(num, -1))
            loss_c = tf.cast(loss_c, tf.float32)
            loss_c = tf.where(condition=pos, x=0.0, y=loss_c)
            loss_c = tf.reshape(loss_c, shape=(num, -1))
            loss_idx = tf.argsort(values=loss_c, axis=1, direction="DESCENDING")
            idx_rank = tf.argsort(values=loss_idx, axis=1, direction="ASCENDING")
            num_pos = tf.reduce_sum(tf.cast(pos, dtype=tf.int32), axis=1, keepdims=True)
            num_neg = clip_by_value(t=self.negpos_ratio * num_pos, clip_value_max=pos.shape[1] - 1)
            neg = idx_rank < num_neg

            # loss
            pos_idx = tf.broadcast_to(tf.expand_dims(pos, axis=2), shape=conf_data_layer.shape)
            neg_idx = tf.broadcast_to(tf.expand_dims(neg, axis=2), shape=conf_data_layer.shape)
            # loss per sample
            ce_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                  reduction=tf.keras.losses.Reduction.SUM)
            idx_p = tf.math.logical_or(pos_idx, neg_idx)
            idx_t = tf.math.logical_or(pos, neg)
            loss_c_list = []
            for j in range(num):
                conf_p = tf.boolean_mask(tensor=conf_data_layer[j, ...], mask=idx_p[j, ...])
                conf_p = tf.reshape(conf_p, shape=(-1, self.num_classes))  # (68380, 21)
                targets_weighted = tf.boolean_mask(tensor=conf_t[j, ...], mask=idx_t[j, ...])  # (68380,)
                if self.loss == 'focal':
                    onehot_labels_partial = tf.one_hot(targets_weighted, self.num_classes)[:, :-1]
                    zero_array = tf.zeros(((self.negpos_ratio * num_pos + num_pos)[j][0], 1))
                    one_hot_labels = tf.concat([onehot_labels_partial, zero_array], axis=1)
                    loss_c = self._focal_loss(one_hot_labels, conf_p)
                else:
                    loss_c = ce_fn(y_true=targets_weighted, y_pred=conf_p)
                loss_c_list.append(loss_c)
            loss_c_tensor = tf.stack(values=loss_c_list, axis=0)
            loss_l = loss_l_tensor / tf.cast(num_pos[:, 0], dtype=tf.float32)
            loss_c = loss_c_tensor / tf.cast(num_pos[:, 0], dtype=tf.float32)
            l_losses.append(loss_l)
            c_losses.append(loss_c)
        return l_losses, c_losses


# ----------------------------------------------------------- MultiBox loss ------------#
# -------------------------------------------- Default Box detectron2 ----------------- #
class DefaultBoxesDetectron2:
    def __init__(self, image_size: Tuple[int, int], feature_maps: Tuple[Tuple[int, int], ...],
                 box_sizes: Tuple[Tuple[float, ...], ...], aspect_ratios: Tuple[float, ...], strides: Tuple[int, ...],
                 offset: int):
        self.image_size = image_size
        self.feature_maps = feature_maps
        self.box_sizes = box_sizes
        self.aspect_ratios = aspect_ratios
        self.strides = strides
        self.offset = offset
        self.anchors = self.generate_cell_anchors()

    def generate_cell_anchors(self):
        """
        This returns anchors, located at (0,0) sized according to to box_sizes.
        :return: np.ndarray of cell_anchors  (len(FEATURE_MAPS), number of anchors, 4) 4: X,Y,W,H
        """
        layer_anchors = []
        for layer_box_size in self.box_sizes:
            anchors = []
            for size in layer_box_size:
                area = size ** 2.0
                for aspect_ratio in self.aspect_ratios:
                    w = math.sqrt(area / aspect_ratio)
                    h = aspect_ratio * w
                    x0, y0 = 0., 0.
                    anchors.append([x0, y0, w, h])
            layer_anchors.append(np.array(anchors))
        return np.stack(layer_anchors)

    def _create_grid_offsets(self, size: Tuple[int, int], stride: Tuple[int, int, int, int, int]):
        grid_height, grid_width = size
        shifts_x = np.arange(self.offset * stride, grid_width * stride, step=stride, dtype=np.float32)
        shifts_y = np.arange(self.offset * stride, grid_height * stride, step=stride, dtype=np.float32)
        shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        return shift_x, shift_y

    def generate_anchors(self):
        """
        Returns:
            list[Tensor]: #featuremap tensors, each is (#locations x #cell_anchors) x 4
        """
        anchors = []
        buffers = self.anchors
        grid_sizes = self.feature_maps
        for size, stride, base_anchors in zip(grid_sizes, self.strides, buffers):
            shift_x, shift_y = self._create_grid_offsets(size, stride)
            shifts = np.stack((shift_x, shift_y, np.zeros_like(shift_x), np.zeros_like(shift_y)), axis=1)
            absolute_anchors = (shifts.reshape(-1, 1, 4) + base_anchors.reshape(1, -1, 4)).reshape(-1, 4)
            normalized_anchors = absolute_anchors / np.array([self.image_size[1], self.image_size[0],
                                                              self.image_size[1], self.image_size[0]])
            anchors.append(normalized_anchors)
        return anchors


# ----------------------------------------- Default Box detectron2--------------------- #
# -------------------------------------OD Functions ----------------------------------- #
BUCKET_NAME = 'example-datasets-47ml982d'
PROJECT_ID = 'example-dev-project-nmrksf0o'
BACKGROUND_LABEL = 79
MAX_BB_PER_IMAGE = 25
PRIORS = 3
CLASSES = 80
CATEGORIES = ['person', 'car']
IMAGE_SIZE = (800, 1067)
FEATURE_MAPS = ((100, 136), (50, 68), (25, 34), (13, 17), (7, 9))
BOX_SIZES = ((32., 40.31747359663594, 50.79683366298238),
             (64., 80.63494719327188, 101.59366732596476),
             (128., 161.26989438654377, 203.18733465192952),
             (256., 322.53978877308754, 406.37466930385904),
             (512., 645.0795775461751, 812.7493386077181))
PIXEL_MEAN = [103.53, 116.28, 123.675]
ASPECT_RATIOS = (0.5, 1, 2)
NUM_PROIRS = len(BOX_SIZES[0]) * len(ASPECT_RATIOS)
NUM_FEATURES = len(FEATURE_MAPS)
OFFSET = 0
STRIDES = (8, 16, 32, 64, 128)
PADDING = 21
LOAD_UNION_CATEGORIES_IMAGES = False
BOXES_GENERATOR = DefaultBoxesDetectron2(image_size=IMAGE_SIZE, feature_maps=FEATURE_MAPS, box_sizes=BOX_SIZES,
                                         aspect_ratios=ASPECT_RATIOS,
                                         strides=STRIDES, offset=OFFSET)
DEFAULT_BOXES = BOXES_GENERATOR.generate_anchors()
LOSS_FN = MultiBoxLossDetectron(num_classes=CLASSES, overlap_thresh=0.5, neg_pos=3,
                                default_boxes=DEFAULT_BOXES, background_label=BACKGROUND_LABEL)
DECODER = DecoderDetectron(CLASSES,
                           background_label=BACKGROUND_LABEL,
                           top_k=10,
                           conf_thresh=0.01,
                           nms_thresh=0.45)
ALL_CATEGORIES = True


def load_mapping(fpath: str) -> Dict[int, int]:
    with open(fpath, 'r') as file:
        mapping_json = json.load(file)
    int_dict = {int(key): value for key, value in mapping_json.items()}
    return int_dict


@lru_cache()
def _connect_to_gcs_and_return_bucket(bucket_name: str) -> Bucket:
    print("connect to GCS")
    gcs_client = storage.Client(project=PROJECT_ID, credentials=AnonymousCredentials())
    return gcs_client.bucket(bucket_name)


def _download(cloud_file_path: str, local_file_path: Optional[str] = None) -> str:
    print("downloading")
    # if local_file_path is not specified saving in home dir
    if local_file_path is None:
        # if os.path.exists("/nfs/"): #on the platform
        #     home_dir = "/nfs/"
        # else:
        home_dir = os.path.expanduser("~")
        local_file_path = os.path.join(home_dir, "Tensorleap_data_1", BUCKET_NAME, cloud_file_path)

    if os.path.exists(local_file_path):
        return local_file_path
    bucket = _connect_to_gcs_and_return_bucket(BUCKET_NAME)
    dir_path = os.path.dirname(local_file_path)
    os.makedirs(dir_path, exist_ok=True)
    blob = bucket.blob(cloud_file_path)
    blob.download_to_filename(local_file_path)
    return local_file_path


def subset_images() -> List[PreprocessResponse]:
    def load_set(coco, load_union=False):
        # get all images containing given categories
        cat_ids = coco.getCatIds(CATEGORIES)  # Fetch class IDs only corresponding to the filterClasses
        if not load_union:
            img_ids = coco.getImgIds(catIds=cat_ids)  # Get all images containing the Category IDs together
        else:
            img_ids = set()
            for cat_id in cat_ids:
                image_ids = coco.getImgIds(catIds=[cat_id])
                img_ids.update(image_ids)
            img_ids = list(img_ids)
        imgs = coco.loadImgs(img_ids)
        return imgs

    data_type = 'train2014'
    ann_file = f'coco/ms-coco/annotations/instances_{data_type}.json'
    fpath = _download(ann_file)
    # initialize COCO api for instance annotations
    print(fpath)
    traincoco = COCO(fpath)
    print(traincoco)
    x_train_raw = load_set(coco=traincoco, load_union=LOAD_UNION_CATEGORIES_IMAGES)
    data_type = 'val2014'
    ann_file = f'coco/ms-coco/annotations/instances_{data_type}.json'
    fpath = _download(ann_file)
    # initialize COCO api for instance annotations
    valcoco = COCO(fpath)
    x_test_raw = load_set(coco=valcoco, load_union=LOAD_UNION_CATEGORIES_IMAGES)
    train_size = min(len(x_train_raw), 5000)
    val_size = min(len(x_test_raw), 5000)
    mapping_file = 'coco/assets/bb/cat_mapping.json'
    fpath = _download(mapping_file)
    catids_mapping = load_mapping(fpath)
    print("end of subset")
    return [
        PreprocessResponse(length=train_size, data={'cocofile': traincoco, 'samples': x_train_raw[:train_size],
                                                    'subdir': 'train2014', 'cat_map': catids_mapping}),
        PreprocessResponse(length=val_size, data={'cocofile': valcoco, 'samples': x_test_raw[:val_size],
                                                  'subdir': 'val2014', 'cat_map': catids_mapping})]


def get_image(idx: int, subset: PreprocessResponse) -> NDArray[float]:
    """
    Returns a BGR image normalized and padded
    """
    print("in get image")
    data = subset.data
    x = data['samples'][idx]
    filepath = f"coco/ms-coco/{data['subdir']}/{x['file_name']}"
    fpath = _download(filepath)
    image = Image.open(fpath)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = np.array(image)[..., ::-1]  # RGB TO BGR
    new_im = Image.fromarray(image).resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
    new_im = np.array(new_im) - PIXEL_MEAN
    padded_im = np.pad(new_im, [(0, 0), (0, PADDING), (0, 0)])
    # rescale
    return padded_im


def get_bb(idx: int, subset: PreprocessResponse) -> NDArray[np.double]:
    """
    returns an array shaped (MAX_BB_PER_IMAGE, 5) where the channel idx is [X,Y,W,H] normalized to [0,1]
    """
    print("in get bb")
    data = subset.data
    x = data['samples'][idx]
    if not ALL_CATEGORIES:
        cat_ids = data['cocofile'].getCatIds(catNms=CATEGORIES)
    else:  # get all categories bb
        cat_ids = []
    ann_ids = data['cocofile'].getAnnIds(imgIds=x['id'], catIds=cat_ids, iscrowd=None)
    anns = data['cocofile'].loadAnns(ann_ids)
    bboxes = np.zeros([MAX_BB_PER_IMAGE, 5])
    max_anns = min(MAX_BB_PER_IMAGE, len(anns))
    height, width = x['height'], x['width']
    for i in range(max_anns):
        ann = anns[i]
        bbox_realigned = [ann['bbox'][0] + ann['bbox'][2] / 2, ann['bbox'][1] + ann['bbox'][3] / 2, *ann['bbox'][2:]]
        bbox_normalized = np.array([bbox_realigned[0] / width, bbox_realigned[1] / height,
                                    bbox_realigned[2] / width, bbox_realigned[3] / height])
        bboxes[i, :4] = bbox_normalized
        bboxes[i, 4] = data['cat_map'][ann['category_id']]
    bboxes[max_anns:, 4] = BACKGROUND_LABEL
    return bboxes


def compute_losses(y_true: tf.Tensor, y_pred: tf.Tensor) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
    """
    Computes the sum of the classification (CE loss) and localization (regression) losses from all heads
    """
    class_list_reshaped, loc_list_reshaped = reshape_output_list(y_pred)  # add batch
    loss_l, loss_c = LOSS_FN(y_true=y_true, y_pred=(loc_list_reshaped, class_list_reshaped))
    return loss_l, loss_c


def od_loss(y_true: tf.Tensor, y_pred: tf.Tensor):  # return batch
    """
    Sums the classification and regression loss
    """
    loss_l, loss_c = compute_losses(y_true, y_pred)
    combined_losses = [l + c for l, c in zip(loss_l, loss_c)]
    sum_loss = tf.reduce_sum(combined_losses, axis=0)
    non_nan_loss = tf.where(tf.math.is_nan(sum_loss), tf.zeros_like(sum_loss), sum_loss)  # LOSS 0 for NAN losses
    return non_nan_loss


def plot_bb_preds_over_image(image: Union[NDArray[float], tf.Tensor], bb_array: Union[NDArray[float], tf.Tensor],
                             iscornercoded: bool = True, bg_label: int = 0):
    """
    Assumes a (X,Y,W,H) Format for the BB text
    bb_array is (CLASSES,TOP_K,PROPERTIES) WHERE PROPERTIES =(conf,xmin,ymin,xmax,ymax)
    """
    fig = Figure()
    ax = fig.gca()
    colors = ['black', 'red', 'blue', 'green', 'yellow', 'purple']
    if not isinstance(bb_array, np.ndarray):
        bb_array = bb_array.numpy()
    # fig, ax = plt.subplots(figsize=(6, 9)
    if not isinstance(image, np.ndarray):
        image = image.numpy()
    ax.imshow(image.astype(int))
    if len(bb_array.shape) == 3:
        bb_array = bb_array.reshape(-1, bb_array.shape[-1])
    for i in range(bb_array.shape[0]):
        if bb_array[i][-1] != bg_label:
            if iscornercoded:
                x, y, w, h = xyxy_to_xywh_format(bb_array[i][1:5])  # FIXED TOM
                # unormalize to image dimensions
            else:
                x, y = bb_array[i][0], bb_array[i][1]
                w, h = bb_array[i][2], bb_array[i][3]
            x = (x - w * 0.5) * image.shape[1]  # X
            y = (y - h * 0.5) * image.shape[0]  # Y
            w = w * image.shape[1]  # X
            h = h * image.shape[0]  # Y
            ax.add_patch(
                patches.Rectangle((x, y), w, h, fill=False, edgecolor=colors[int(bb_array[i][-1]) % len(colors)],
                                  lw=2))  # FIX TOM identation
    ax.axis('off')
    return fig


def place_holder(idx: int, subset: PreprocessResponse) -> NDArray[np.double]:
    # rescale
    return np.zeros(3)


def unflatten_prediction(keras_output: Union[tf.Tensor, NDArray[float]], num_feature_maps: int,
                         feature_maps: Tuple[Tuple[int, int], ...],
                         feature_channels: int, num_classes: int, num_priors: int, rehsape_fn: Callable):
    """
    keras output - a flat array of a single batch
    """
    output_list = []
    batch = tf.shape(keras_output)[0]
    j = 0
    output_list.append(keras_output[:, :3])
    j += 3
    for i in range(num_feature_maps):
        num_elements = feature_maps[i][0] * feature_maps[i][1] * feature_channels
        output_list.append(rehsape_fn(keras_output[:, j:j + num_elements], (batch, *feature_maps[i], feature_channels)))
        j += num_elements
    for k in range(num_feature_maps):
        # add classes prediction
        num_elements = feature_maps[k][0] * feature_maps[k][1] * num_classes * num_priors
        output_list.append(
            rehsape_fn(keras_output[:, j:j + num_elements], (batch, *feature_maps[k], num_classes * num_priors)))
        j += num_elements
        # add location prediction
        num_elements = feature_maps[k][0] * feature_maps[k][1] * 4 * num_priors
        output_list.append(rehsape_fn(keras_output[:, j:j + num_elements], (batch, *feature_maps[k], 4 * num_priors)))
        j += num_elements
    return output_list


def reshape_output_list(keras_output: tf.Tensor, feature_channels: int = 256):
    """
    reshape the mode's output to two lists sized [NUM_FEATURES] following detectron2 convention.
    class_list item: (BATCH_SIZE, NUM_ANCHORS, CLASSES)
    loc_list item:  (BATCH_SIZE, NUM_ANCHORS, 4)
    """
    rehsape_fn = np.reshape if isinstance(keras_output, np.ndarray) else tf.reshape
    keras_output = unflatten_prediction(keras_output, NUM_FEATURES, FEATURE_MAPS, feature_channels, CLASSES,
                                        NUM_PROIRS, rehsape_fn)
    class_pred_list = keras_output[1 + NUM_FEATURES::2]
    class_list_reshaped = [rehsape_fn(class_pred_list[i], (class_pred_list[i].shape[0], -1, CLASSES))
                           for i in range(len(class_pred_list))]
    loc_pred_list = keras_output[2 + NUM_FEATURES::2]
    loc_list_reshaped = [rehsape_fn(loc_pred_list[i], (loc_pred_list[i].shape[0], -1, 4))
                         for i in range(len(loc_pred_list))]
    return class_list_reshaped, loc_list_reshaped


def unnormalize_image(image):
    return image[..., :-PADDING, :] + PIXEL_MEAN


def image_decoder(image):
    return LeapImage(unnormalize_image(image)[..., ::-1].astype('float32'))


def resized_image_heatmap_visualizer(image):  # image is (H,W,1)
    return image[:, :-PADDING, :]


def resized_bb_heatmap_visualizer(image, predictions):  # image is (H,W,1)
    return image[:, :-PADDING, :]


def resized_bb_gt_heatmap_visualizer(image, ground_truth):  # have to duplicate due to parameter name change
    return image[:, :-PADDING, :]


def classification_metric(y_true, y_pred):  # return batch
    _, loss_c = compute_losses(y_true, y_pred)
    return tf.reduce_sum(loss_c, axis=0)


def regression_metric(y_true, y_pred):  # return batch
    loss_l, _ = compute_losses(y_true, y_pred)
    return tf.reduce_sum(loss_l, axis=0)  # shape of batch


def number_of_bb(index: int, subset: PreprocessResponse) -> int:
    bbs = get_bb(index, subset)
    number_of_bb = np.count_nonzero(bbs[..., -1] != BACKGROUND_LABEL)
    return number_of_bb


def avg_bb_area_metadata(index: int, subset: PreprocessResponse) -> float:
    print("in get area bb metadta")
    bbs = get_bb(index, subset)  # x,y,w,h
    valid_bbs = bbs[bbs[..., -1] != BACKGROUND_LABEL]
    areas = valid_bbs[:, 2] * valid_bbs[:, 3]
    print(areas)
    print(areas.mean())
    return areas.mean()


def avg_bb_aspect_ratio(index: int, subset: PreprocessResponse) -> float:
    print("in get aspect ratio bb metadta")
    bbs = get_bb(index, subset)
    valid_bbs = bbs[bbs[..., -1] != BACKGROUND_LABEL]
    assert ((valid_bbs[:, 3] > 0).all())
    aspect_ratios = valid_bbs[:, 2] / valid_bbs[:, 3]
    print(aspect_ratios)
    print(aspect_ratios.mean())
    return aspect_ratios.mean()


def is_class_exist_gen(class_id: int) -> Callable[[int, PreprocessResponse], float]:
    def func(index: int, subset: PreprocessResponse):
        bbs = get_bb(index, subset)
        is_i_exist = (bbs[..., -1] == class_id).any()
        return float(is_i_exist)

    func.__name__ = f'metadata_{class_id}_instances_count'
    return func


def bb_array_to_object(bb_array: Union[NDArray[float], tf.Tensor], iscornercoded: bool = True, bg_label: int = 0,
                       is_gt=False) -> List[BoundingBox]:
    """
    Assumes a (X,Y,W,H) Format for the BB text
    bb_array is (CLASSES,TOP_K,PROPERTIES) WHERE PROPERTIES =(conf,xmin,ymin,xmax,ymax)
    """

    bb_list = []
    if not isinstance(bb_array, np.ndarray):
        bb_array = bb_array.numpy()
    # fig, ax = plt.subplots(figsize=(6, 9)
    if len(bb_array.shape) == 3:
        bb_array = bb_array.reshape(-1, bb_array.shape[-1])
    for i in range(bb_array.shape[0]):
        if bb_array[i][-1] != bg_label:
            if iscornercoded:
                x, y, w, h = xyxy_to_xywh_format(bb_array[i][1:5])  # FIXED TOM
                # unormalize to image dimensions
            else:
                x, y = bb_array[i][0], bb_array[i][1]
                w, h = bb_array[i][2], bb_array[i][3]
            conf = 1 if is_gt else bb_array[i][0]
            curr_bb = BoundingBox(x=x, y=y, width=w, height=h, confidence=conf, label=cat_mapping[bb_array[i][-1] + 1])
            bb_list.append(curr_bb)
    return bb_list


def gt_decoder(image, ground_truth) -> LeapImageWithBBox:
    bb_object: List[BoundingBox] = bb_array_to_object(ground_truth, iscornercoded=False, bg_label=BACKGROUND_LABEL,
                                                      is_gt=True)
    return LeapImageWithBBox(data=unnormalize_image(image)[..., ::-1].astype(np.float32), bounding_boxes=bb_object)


def bb_decoder(image, predictions):
    """
    Overlays the BB predictions on the image
    """
    class_list_reshaped, loc_list_reshaped = reshape_output_list(
        np.reshape(predictions, (1, predictions.shape[0])))  # add batch
    outputs = DECODER(loc_list_reshaped,
                      class_list_reshaped,
                      DEFAULT_BOXES,
                      from_logits=True
                      )
    bb_object = bb_array_to_object(outputs[0], iscornercoded=True, bg_label=BACKGROUND_LABEL)
    # fig = plot_bb_preds_over_image(unnormalize_image(image)[...,::-1], outputs[0], bg_label=BACKGROUND_LABEL)
    # np_image = matplotlib_to_numpy(fig)
    return LeapImageWithBBox(data=unnormalize_image(image)[..., ::-1].astype(np.float32), bounding_boxes=bb_object)


# Tensorleap's default
leap_binder.set_visualizer(gt_decoder, 'gt_decoder', LeapDataType.ImageWithBBox,
                           heatmap_visualizer=resized_bb_gt_heatmap_visualizer)
leap_binder.set_visualizer(bb_decoder, 'bb_decoder', LeapDataType.ImageWithBBox,
                           heatmap_visualizer=resized_bb_heatmap_visualizer)

# Tom's visualizers
leap_binder.set_visualizer(image_decoder, 'image_decoder', LeapDataType.Image,
                           heatmap_visualizer=resized_image_heatmap_visualizer)
leap_binder.set_preprocess(subset_images)
leap_binder.set_input(place_holder, 'place_holder')
leap_binder.set_input(get_image, 'images')
leap_binder.set_ground_truth(get_bb, 'gt')
ALLCATS = ["prior_" + str(i) + "_" + "class_" + str(j) for j in range(CLASSES) for i in range(PRIORS)]
ALLLOCS = ["prior_" + str(i) + "_" + "loc_" + str(j) for i in range(PRIORS) for j in range(4)]
ALLLABELS = ALLCATS + ALLLOCS


leap_binder.set_metadata(number_of_bb, DatasetMetadataType.int, 'bb_count')
leap_binder.set_metadata(avg_bb_aspect_ratio, DatasetMetadataType.float, 'avg_bb_aspect_ratio')
leap_binder.set_metadata(avg_bb_area_metadata, DatasetMetadataType.float, 'avg_bb_area')
for i in range(4):
    leap_binder.set_metadata(is_class_exist_gen(i), DatasetMetadataType.float, f'does_{i}_exist')
# TODO:
# -------------------------------------------
leap_binder.add_custom_loss(od_loss, 'od_loss')
#-----------------------------------


leap_binder.add_prediction('flattened prediction', ["label"], [],
                           [regression_metric, classification_metric])  # [Metric.MeanIOU]U

cat_mapping = {0: u'__background__',
               1: u'person',
               2: u'bicycle',
               3: u'car',
               4: u'motorcycle',
               5: u'airplane',
               6: u'bus',
               7: u'train',
               8: u'truck',
               9: u'boat',
               10: u'traffic light',
               11: u'fire hydrant',
               12: u'stop sign',
               13: u'parking meter',
               14: u'bench',
               15: u'bird',
               16: u'cat',
               17: u'dog',
               18: u'horse',
               19: u'sheep',
               20: u'cow',
               21: u'elephant',
               22: u'bear',
               23: u'zebra',
               24: u'giraffe',
               25: u'backpack',
               26: u'umbrella',
               27: u'handbag',
               28: u'tie',
               29: u'suitcase',
               30: u'frisbee',
               31: u'skis',
               32: u'snowboard',
               33: u'sports ball',
               34: u'kite',
               35: u'baseball bat',
               36: u'baseball glove',
               37: u'skateboard',
               38: u'surfboard',
               39: u'tennis racket',
               40: u'bottle',
               41: u'wine glass',
               42: u'cup',
               43: u'fork',
               44: u'knife',
               45: u'spoon',
               46: u'bowl',
               47: u'banana',
               48: u'apple',
               49: u'sandwich',
               50: u'orange',
               51: u'broccoli',
               52: u'carrot',
               53: u'hot dog',
               54: u'pizza',
               55: u'donut',
               56: u'cake',
               57: u'chair',
               58: u'couch',
               59: u'potted plant',
               60: u'bed',
               61: u'dining table',
               62: u'toilet',
               63: u'tv',
               64: u'laptop',
               65: u'mouse',
               66: u'remote',
               67: u'keyboard',
               68: u'cell phone',
               69: u'microwave',
               70: u'oven',
               71: u'toaster',
               72: u'sink',
               73: u'refrigerator',
               74: u'book',
               75: u'clock',
               76: u'vase',
               77: u'scissors',
               78: u'teddy bear',
               79: u'hair drier',
               80: u'toothbrush'}