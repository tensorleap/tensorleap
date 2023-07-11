from typing import List, Dict
from pathlib import Path

from domain_gap.utils.configs import VAL_INDICES


def get_kitti_data() -> Dict[str, List[str]]:
    dataset_path = Path('KITTI/data_semantics/training')
    train_indices = [i for i in range(200) if i not in VAL_INDICES]
    indices_lists = [train_indices, VAL_INDICES]
    data_dict = {'train': {}, "validation": {}}
    TRAIN_SIZE = 170
    VAL_SIZE = 30
    for indices, size, title in zip(indices_lists, (TRAIN_SIZE, VAL_SIZE), ("train", "validation")):
        images = [str(dataset_path / "image_2" / (str(i).zfill(6) + "_10.png")) for i in indices]
        gt_labels = [str(dataset_path / "semantic" / (str(i).zfill(6) + "_10.png")) for i in indices]
        gt_images = [str(dataset_path / "semantic_rgb" / (str(i).zfill(6) + "_10.png")) for i in indices]
        data_dict[title]['image_path'] = images
        data_dict[title]['gt_path'] = gt_labels
        data_dict[title]['gt_image_path'] = gt_images
    return data_dict
