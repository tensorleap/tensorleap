from cityscapes.gcs_utils import _connect_to_gcs_and_return_bucket, BUCKET_NAME
import os
from typing import Optional
import json

from cityscapes.preprocessing import load_cityscapes_data
from cityscapes.utils.general_utils import extract_bounding_boxes_from_instance_segmentation_polygons


def _download(cloud_file_path: str, set_name, kind, local_file_path: Optional[str] = None) -> str:
    # if local_file_path is not specified saving in home dir
    if local_file_path is None:
        home_dir = os.getenv("HOME")
        name = cloud_file_path.split('/')
        if kind == 'images':
            local_file_path = os.path.join(home_dir, "Tensorleap_data_Cityscapes", kind, set_name, name[-1][:-16]+'.png')
        else:
            local_file_path = os.path.join(home_dir, "Tensorleap_data_Cityscapes", kind, set_name, name[-1][:-21]+'.json')

    # check if file already exists
    if os.path.exists(local_file_path):
        return local_file_path
    bucket = _connect_to_gcs_and_return_bucket(BUCKET_NAME)
    dir_path = os.path.dirname(local_file_path)
    os.makedirs(dir_path, exist_ok=True)
    blob = bucket.blob(cloud_file_path)
    blob.download_to_filename(local_file_path)
    return local_file_path

def image_save(image, set_name, kind):
    cloud_image_path = image
    _download(cloud_image_path, set_name, kind)

def label_save(gt_label, set_name, kind):
    cloud_label_path = gt_label
    fpath = _download(cloud_label_path, set_name, kind)
    with open(fpath, 'r') as file:
        json_data = json.load(file)
    bounding_boxes = extract_bounding_boxes_from_instance_segmentation_polygons(json_data)
    file_object = open(f"{fpath[:-4]}txt", "w")
    for bb in bounding_boxes:
        # bb representation: (x, y, width, height)
        file_object.write(f"{int(bb[4])} {bb[0]} {bb[1]} {bb[2]} {bb[3]}\n")
    file_object.close()

if __name__=='__main__':
    all_images, all_gt_images, all_gt_labels, all_gt_labels_for_bbx, all_file_names, all_metadata, all_cities = load_cityscapes_data()
    for i, set_name in enumerate(['train', 'val', 'test']):
        for (image, gt_label) in zip(all_images[i], all_gt_labels_for_bbx[i]):
            image_save(image, set_name, 'images')
            label_save(gt_label, set_name, 'labels')






