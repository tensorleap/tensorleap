from typing import List, Dict
from code_loader.contract.datasetclasses import PreprocessResponse

from domain_gap.data.cs_data import get_cityscapes_data
from domain_gap.data.kitti_data import get_kitti_data
from domain_gap.utils.configs import TRAIN_SIZE, VAL_SIZE


def subset_images() -> List[PreprocessResponse]:
    subset_sizes = [TRAIN_SIZE, VAL_SIZE]
    cs_responses: List[PreprocessResponse] = get_cityscapes_data()
    kitti_data: Dict[str, List[str]] = get_kitti_data()
    sub_names = ["train", "validation"]
    for i, title in enumerate(sub_names):   # add kitti values
        cs_responses[i].data['image_path'] += kitti_data[title]['image_path']
        cs_responses[i].data['gt_path'] += kitti_data[title]['gt_path']
        cs_responses[i].data['file_names'] += kitti_data[title]['image_path']
        cs_responses[i].data['cities'] += ["Karlsruhe"] * len(kitti_data[title]['image_path'])
        cs_responses[i].data['dataset'] += ['kitti'] * len(kitti_data[title]['image_path'])
        cs_responses[i].data['real_size'] += len(kitti_data[title]['image_path'])
        cs_responses[i].data['metadata'] += [""] * len(kitti_data[title]['image_path'])
        cs_responses[i].length += len(kitti_data[title]['image_path'])
        cs_responses[i].length = subset_sizes[i]
    return cs_responses