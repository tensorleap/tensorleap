import json
from typing import Dict
import numpy as np
from code_loader.contract.datasetclasses import PreprocessResponse
from PIL import Image

from domain_gap.utils.gcs_utils import _download
from domain_gap.data.cs_data import Cityscapes
from domain_gap.utils.configs import IMAGE_SIZE, AUGMENT, TRAIN_SIZE


def get_categorical_mask(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data
    cloud_path = data['gt_path'][idx % data["real_size"]]
    fpath = _download(cloud_path)
    mask = np.array(Image.open(fpath).resize(IMAGE_SIZE, Image.Resampling.NEAREST))
    if data['dataset'][idx % data["real_size"]] == 'cityscapes':
        encoded_mask = Cityscapes.encode_target_cityscapes(mask)
    else:
        encoded_mask = Cityscapes.encode_target(mask)
    return encoded_mask


def get_metadata_json(idx: int, data: PreprocessResponse) -> Dict[str, str]:
    cloud_path = data.data['metadata'][idx]
    fpath = _download(cloud_path)
    with open(fpath, 'r') as f:
        metadata_dict = json.loads(f.read())
    return metadata_dict


def aug_factor_or_zero(idx: int, data: PreprocessResponse, value: float) -> float:
    if data.data["subset_name"] == "train" and AUGMENT and idx > TRAIN_SIZE - 1:
        return value.numpy()
    else:
        return 0.