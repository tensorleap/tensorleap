import tensorflow as tf
import numpy as np


def make_color_wheel() -> np.ndarray:
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(
        np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(
        np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(
        np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(
        np.floor(255 * np.arange(0, BM) / BM))
    col += +BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(
        np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel


COLORWHEEL = make_color_wheel()


def flow_to_image(flow: tf.Tensor, maxrad: int=-1, unknown_flow_thresh: int=1e7) -> np.ndarray:
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    idxUnknow = (abs(u) > unknown_flow_thresh) | (abs(v) > unknown_flow_thresh)
    u[idxUnknow] = 0
    v[idxUnknow] = 0
    if maxrad == -1:
        rad = np.sqrt(u**2 + v**2)
        maxrad = max(-1, np.max(rad))
    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)
    img = compute_color(u, v)
    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def decode_kitti_png(flow_img: np.ndarray) -> np.ndarray:
    """
    :param flow_img:
    :return: decoded opt flow gt such that:
        1st channel is x movement,
        2nd channel is y movement,
        3rd channel is pixels mask
    """
    flow_img = flow_img.astype(np.float32)
    flow_data = np.zeros(flow_img.shape, dtype=np.float32)
    flow_data[:, :, 0] = (flow_img[:, :, 2] - 2 ** 15) / 64.0
    flow_data[:, :, 1] = (flow_img[:, :, 1] - 2 ** 15) / 64.0
    flow_data[:, :, 2] = flow_img[:, :, 0]
    return flow_data


def compute_color(u: tf.Tensor, v: tf.Tensor) -> np.ndarray:
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    ncols = np.size(COLORWHEEL, 0)
    rad = np.sqrt(u**2 + v**2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0
    for i in range(0, np.size(COLORWHEEL, 1)):
        tmp = COLORWHEEL[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))
    return img


def get_flow_magnitude(flow: tf.Tensor) -> float:
    return tf.sqrt(flow[..., 0]**2+flow[..., 1]**2)


def EPE_mask(gt_flow:tf.Tensor, pred_flow:tf.Tensor) -> tf.Tensor:
    mask = gt_flow[..., 2]
    gt_flow = gt_flow[..., 0:2]
    pixel_err = tf.sqrt(tf.square(gt_flow[..., 0] - pred_flow[..., 0]) + tf.square(gt_flow[..., 1] - pred_flow[..., 1]))
    pixel_err = pixel_err * mask
    return pixel_err


def get_fl_map(gt_flow: tf.Tensor, pred_flow: tf.Tensor) -> tf.Tensor:
    res = EPE_mask(gt_flow, pred_flow)
    fl_map = (res > 3) & (res/(get_flow_magnitude(gt_flow)+10**(-5)) > 0.05)
    return fl_map

