import cv2
from typing import List, Dict, Union
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse
import tensorflow as tf
from PIL import Image
import numpy as np
import numpy.typing as npt
import json
from code_loader.contract.visualizer_classes import LeapImage
from code_loader.contract.enums import (
    LeapDataType
)

from optical_flow_raft.config import BUCKET_NAME, MAX_SCENE, MAX_STEREO, IMG_SIZE
from optical_flow_raft.data.preprocess import get_kitti_data
from optical_flow_raft.utils.flow_utils import decode_kitti_png, flow_to_image, EPE_mask, get_fl_map
from optical_flow_raft.utils.gcs_utils import download

# --------------------------------------------------inputs & GT---------------------------------------------------------


def subset_images() -> List[PreprocessResponse]:
    scene_flow = get_kitti_data(bucket_name=BUCKET_NAME, data_subset="scene")
    stereo_flow = get_kitti_data(bucket_name=BUCKET_NAME, data_subset="stereo")
    scene_flow_poe_p = download("KITTI/data_scene_flow/estimated_poe/scene_flow_poe.json", bucket_name=BUCKET_NAME)
    with open(scene_flow_poe_p, 'r') as f:
        scene_flow_poe = json.load(f)
    stereo_flow_poe_p = download("KITTI/data_stereo_flow/estimated_poe/combined_poe.json", bucket_name=BUCKET_NAME)
    with open(stereo_flow_poe_p, 'r') as f:
        stereo_flow_poe = json.load(f)
    responses = [
                 PreprocessResponse(length=min(len(scene_flow.train_IDs), MAX_SCENE),
                                    data={"dataset_name": "scene_flow", 'paths': scene_flow.train_IDs,
                                          'poe': scene_flow_poe}),
                 PreprocessResponse(length=min(len(stereo_flow.train_IDs), MAX_STEREO),
                                    data={"dataset_name": "stereo_flow", 'paths': stereo_flow.train_IDs,
                                          'poe': stereo_flow_poe})
                 ]
    return responses


def get_image(cloud_path: str) -> np.ndarray:
    fpath = download(str(cloud_path), bucket_name=BUCKET_NAME)
    flow_img = cv2.imread(fpath, -1)
    flow_img = cv2.resize(flow_img, IMG_SIZE[::-1])
    return flow_img


def input_image1(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data
    path0 = data['paths'][idx][0]
    img0 = get_image(path0)
    return img0


def input_image2(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data
    path1 = data['paths'][idx][1]
    img1 = get_image(path1)
    return img1


def gt_encoder(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data
    path = data['paths'][idx][2]
    img = get_image(path)
    img = decode_kitti_png(img)
    return img
# -------------------------------------------------------- metadata ----------------------------------------------------


def masked_of_percent(idx: int, data: PreprocessResponse) -> float:
    gt = gt_encoder(idx, data)
    return 100*gt[..., -1].sum()/(gt.shape[0]*gt.shape[1])


def metadata_filename(idx: int, data: PreprocessResponse) -> str:
    data = data.data
    path = data['paths'][idx][0]
    fname = path.split('/')[-1]
    return fname


def dataset_name(idx: int, subset: PreprocessResponse) -> str:
    return subset.data['dataset_name']


def metadata_idx(idx: int, data: PreprocessResponse) -> int:
    return idx


def average_of_magnitude(idx: int, data: PreprocessResponse):
    gt = gt_encoder(idx,data)
    return np.mean(np.sqrt(gt[gt[..., -1].astype(bool),0]**2+(gt[gt[..., -1].astype(bool), 1])**2))


def poe_x(idx: int, data: PreprocessResponse) -> float:
    filename = metadata_filename(idx, data)
    return data.data['poe'][filename][0]


def poe_y(idx: int, data: PreprocessResponse) -> float:
    filename = metadata_filename(idx, data)
    return data.data['poe'][filename][1]


def mu_over_sigma_of(idx: int, data: PreprocessResponse):
    gt = gt_encoder(idx, data)
    x_gt = gt[gt[..., -1] != 0, 0]
    y_gt = gt[gt[..., -1] != 0, 1]
    out_angles = np.divide(y_gt, x_gt, out=np.zeros_like(x_gt), where=x_gt != 0)
    std = out_angles.std()
    mean = out_angles.mean()
    if std > 0:
        return mean/std
    else:
        return mean

def metadata_dict(idx: int, data: PreprocessResponse) -> Dict[str, Union[float, int, str]]:
    metadata_functions = {
        "masked_of_percent": masked_of_percent,
        "metadata_filename": metadata_filename,
        "dataset_name": dataset_name,
        "metadata_idx": metadata_idx,
        "average_of_magnitude": average_of_magnitude,
        "poe_x": poe_x,
        "poe_y": poe_y,
        "mu_over_sigma_of": mu_over_sigma_of
    }

    res = dict()
    for func_name, func in metadata_functions.items():
        res[func_name] = func(idx, data)
    return res
# -------------------------------------------------------- visualizers -------------------------------------------------


def image_visualizer(image: npt.NDArray[np.float32]) -> LeapImage:
    return LeapImage(image[..., ::-1])


def flow_visualizer(flow: npt.NDArray[np.float32]) -> LeapImage:
    img = flow_to_image(flow)
    return LeapImage(img)


def gt_visualizer(flow: npt.NDArray[np.float32]) -> LeapImage:
    img = flow_to_image(flow)
    img[(img == 255).all(axis=-1)] = 0
    return LeapImage(img)


def mask_visualizer(mask: npt.NDArray[np.uint8]) -> LeapImage:
    return LeapImage((mask[..., None].repeat(3,axis=2)*255).astype(np.uint8))

# -------------------------------------------------------- metrics  -------------------------------------------------


def EPE(gt_flow: tf.Tensor, pred_flow: tf.Tensor) -> tf.Tensor:
    # gt_flow shape: (batch_size, height, width, 2)
    # pred_flow shape: (batch_size, height, width, 2)
    # v_gt - v_pred u_gt - u_pred
    pixel_err = EPE_mask(gt_flow, pred_flow)
    sample_err = tf.reduce_mean(pixel_err, axis=[1, 2])
    return sample_err


def fg_mask(idx: int, subset: PreprocessResponse) -> np.ndarray:
    if subset.data['dataset_name'] == 'scene_flow':
        filename = metadata_filename(idx, subset)
        local_file = download(f"KITTI/data_scene_flow/training/obj_map/{filename}", bucket_name=BUCKET_NAME)
        return np.array(Image.open(local_file).resize(IMG_SIZE[::-1], Image.NEAREST)).astype(np.float32)
    elif subset.data['dataset_name'] == 'stereo_flow':
        return np.ones_like(input_image1(idx, subset)[..., 0]).astype(np.float32)


def fl_metric(gt_flow: tf.Tensor, pred_flow: tf.Tensor) -> tf.Tensor:
    fl_map = get_fl_map(gt_flow, pred_flow)
    outliers_num = tf.math.count_nonzero(fl_map, axis=[1, 2])
    return outliers_num / (tf.maximum(tf.math.count_nonzero(gt_flow[..., -1], axis=[1, 2]), 1))


def fl_foreground(gt_flow: tf.Tensor, pred_flow: tf.Tensor, foreground_map: tf.Tensor) -> tf.Tensor:
    fl_map = tf.cast(get_fl_map(gt_flow, pred_flow), float)*foreground_map
    outliers_num = tf.math.count_nonzero(fl_map, axis=[1, 2])
    combined_mask = gt_flow[..., -1]*foreground_map
    return outliers_num / (tf.maximum(tf.math.count_nonzero(combined_mask, axis=[1, 2]), 1))


def fl_background(gt_flow: tf.Tensor, pred_flow: tf.Tensor, foreground_map: tf.Tensor) -> tf.Tensor:
    background_mask = 1 - foreground_map
    fl_map = tf.cast(get_fl_map(gt_flow, pred_flow), float)*background_mask
    outliers_num = tf.math.count_nonzero(fl_map, axis=[1, 2])
    combined_mask = gt_flow[..., -1]*background_mask
    return outliers_num /(tf.maximum(tf.math.count_nonzero(combined_mask, axis=[1, 2]), 1))

# -------------------------------------------------------- binding  -------------------------------------------------
#preprocess function
leap_binder.set_preprocess(subset_images)

#set input and gt
leap_binder.set_input(input_image1, 'image1')
leap_binder.set_input(input_image2, 'image2')
leap_binder.set_input(fg_mask, 'fg_mask')
leap_binder.set_ground_truth(gt_encoder, 'mask')

#set prediction
leap_binder.add_prediction('opt_flow', ['x', 'y'], metrics=[])

#set meata_data
leap_binder.set_metadata(metadata_dict, name='metadata')

#set visualizer
leap_binder.set_visualizer(image_visualizer, 'image_visualizer', LeapDataType.Image)
leap_binder.set_visualizer(flow_visualizer, 'flow_visualizer', LeapDataType.Image)
leap_binder.set_visualizer(gt_visualizer, 'gt_visualizer', LeapDataType.Image)
leap_binder.set_visualizer(mask_visualizer, 'fg_visualizer', LeapDataType.Image)

#set loss
leap_binder.add_custom_loss(EPE, 'EPE')

# set custom metrics
leap_binder.add_custom_metric(fl_metric, 'FL-all')
leap_binder.add_custom_metric(fl_foreground, 'FL-fg')
leap_binder.add_custom_metric(fl_background, 'FL-bg')
