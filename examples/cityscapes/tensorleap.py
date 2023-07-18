from typing import List, Optional, Dict, Callable

import tensorflow as tf


from PIL import Image
import numpy as np
import numpy.typing as npt
import json
from matplotlib import colors
from matplotlib import cm as cmx
import matplotlib.pyplot as plt

from cityscapes.gcs_utils import _download
from cityscapes.metrics import regression_metric, classification_metric, object_metric
from cityscapes.preprocessing import IMAGE_STD, IMAGE_MEAN, categories, Cityscapes, image_size, load_cityscapes_data, \
    BACKGROUND_LABEL
from cityscapes.utils.general_utils import polygon_to_bbox

from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.visualizer_classes import LeapImage
from code_loader.contract.enums import (
    DatasetMetadataType,
    LeapDataType
)

from cityscapes.visualizers.visualizers import bb_decoder, gt_bb_decoder


# ----------------------------------------------------data processing--------------------------------------------------
def load_cityscapes_data_leap() -> List[PreprocessResponse]:
    all_images, all_gt_images, all_gt_labels, all_gt_labels_for_bbx, all_file_names, all_metadata, all_cities =\
        load_cityscapes_data()

    responses = [PreprocessResponse(length=len(all_images[0]), data={
                "image_path": all_images[0],
                "subset_name": "train",
                "gt_path": all_gt_labels[0],
                "gt_bbx_path": all_gt_labels_for_bbx[0],
                "gt_image_path": all_gt_images[0],
                "real_size": len(all_images[0]),
                "file_names": all_file_names[0],
                "cities": all_cities[0],
                "metadata": all_metadata[0],
                "dataset": ["cityscapes"]*len(all_images[0])}),
                PreprocessResponse(length=len(all_images[1]), data={
                "image_path": all_images[1],
                "subset_name": "val",
                "gt_path": all_gt_labels[1],
                "gt_bbx_path": all_gt_labels_for_bbx[0],
                "gt_image_path": all_gt_images[1],
                "real_size": len(all_images[1]),
                "file_names": all_file_names[1],
                "cities": all_cities[1],
                "metadata": all_metadata[1],
                "dataset": ["cityscapes"]*len(all_images[1])})]
    return responses



def non_normalized_image(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data
    cloud_path = data['image_path'][idx%data["real_size"]]
    fpath = _download(str(cloud_path))
    img = np.array(Image.open(fpath).convert('RGB').resize(image_size))/255.
    return img


def input_image(idx: int, data: PreprocessResponse) -> np.ndarray:
    img = non_normalized_image(idx%data.data["real_size"], data)
    normalized_image = (img - IMAGE_MEAN)/IMAGE_STD
    return normalized_image.astype(float)



def extract_bounding_boxes_from_instance_segmentation_polygons(json_data):
    objects = json_data['objects']
    bounding_boxes = []
    image_size = (json_data['imgHeight'], json_data['imgWidth'])
    for object in objects:
        b = np.zeros(5)
        class_label = object['label']
        class_id = Cityscapes.get_class_id(class_label) #TODO: class label to class id
        bbox = polygon_to_bbox(object['polygon'])
        bbox /= np.array((image_size[1], image_size[0], image_size[1], image_size[0]))
        b[:4] = bbox
        b[4] = class_id
        bounding_boxes.append(b)
    return bounding_boxes

def ground_truth_bbox(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data
    cloud_path = data['gt_bbx_path'][idx%data["real_size"]]
    fpath = _download(cloud_path)
    with open(fpath, 'r') as file:
        json_data = json.load(file)
    bounding_boxes = extract_bounding_boxes_from_instance_segmentation_polygons(json_data)
    return bounding_boxes

def number_of_bb(index: int, subset: PreprocessResponse) -> int:
    bbs = ground_truth_bbox(index, subset)
    number_of_bb = np.count_nonzero(bbs[..., -1] != BACKGROUND_LABEL)
    return number_of_bb

def avg_bb_aspect_ratio(index: int, subset: PreprocessResponse) -> float:
    bbs = ground_truth_bbox(index, subset)
    valid_bbs = bbs[bbs[..., -1] != BACKGROUND_LABEL]
    assert ((valid_bbs[:, 3] > 0).all())
    aspect_ratios = valid_bbs[:, 2] / valid_bbs[:, 3]
    return aspect_ratios.mean()

def avg_bb_area_metadata(index: int, subset: PreprocessResponse) -> float:
    bbs = ground_truth_bbox(index, subset)  # x,y,w,h
    valid_bbs = bbs[bbs[..., -1] != BACKGROUND_LABEL]
    areas = valid_bbs[:, 2] * valid_bbs[:, 3]
    return areas.mean()

def is_class_exist_gen(class_id: int) -> Callable[[int, PreprocessResponse], float]:
    def func(index: int, subset: PreprocessResponse):
        bbs = ground_truth_bbox(index, subset)
        is_i_exist = (bbs[..., -1] == class_id).any()
        return float(is_i_exist)

    func.__name__ = f'metadata_{class_id}_instances_count'
    return func

# ----------------------------------------------------------metadata----------------------------------------------------

def metadata_filename(idx: int, data: PreprocessResponse) -> str:
    return data.data['file_names'][idx]

def metadata_city(idx: int, data: PreprocessResponse) -> str:
    return data.data['cities'][idx]

def metadata_idx(idx: int, data: PreprocessResponse) -> int:
    return idx

def get_metadata_json(idx: int, data: PreprocessResponse) -> Dict[str,str]:
    cloud_path = data.data['metadata'][idx]
    fpath = _download(cloud_path)
    with open(fpath, 'r') as f:
        metadata_dict = json.loads(f.read())
    return metadata_dict

def metadata_gps_heading(idx: int, data: PreprocessResponse) -> float:
    return get_metadata_json(idx, data)['gpsHeading']

def metadata_gps_latitude(idx: int, data: PreprocessResponse) -> float:
    return get_metadata_json(idx, data)['gpsLatitude']

def metadata_gps_longtitude(idx: int, data: PreprocessResponse) -> float:
    return get_metadata_json(idx, data)['gpsLongitude']

def metadata_outside_temperature(idx: int, data: PreprocessResponse) -> float:
    return get_metadata_json(idx, data)['outsideTemperature']

def metadata_speed(idx: int, data: PreprocessResponse) -> float:
    return get_metadata_json(idx, data)['speed']

def metadata_yaw_rate(idx: int, data: PreprocessResponse) -> float:
    return get_metadata_json(idx, data)['yawRate']

#todo: THINK ON METADATA

# def metadata_brightness(idx: int, data: PreprocessResponse) -> float:
#     img = non_normalized_image(idx%data.data["real_size"], data)
#     return np.mean(img)


# def aug_factor_or_zero(idx: int, data: PreprocessResponse, value: float) -> float:
#     if data.data["subset_name"] == "train" and AUGMENT and idx > TRAIN_SIZE-1:
#         return value.numpy()
#     else:
#         return 0.

# def metadata_hue_factor(idx: int, data: PreprocessResponse) -> float:
#     factor = stateless_random_uniform(shape=[], minval=-HUE_LIM, maxval=HUE_LIM, seed=(idx, idx+1))
#     return aug_factor_or_zero(idx, data, factor)

# def metadata_saturation_factor(idx: int, data: PreprocessResponse) -> float:
#     factor = stateless_random_uniform(shape=[], minval=1-SATUR_LIM, maxval=1+SATUR_LIM, seed=(idx, idx+1))
#     return aug_factor_or_zero(idx, data, factor)


# def metadata_contrast_factor(idx: int, data: PreprocessResponse) -> float:
#     factor = stateless_random_uniform(shape=[], minval=1-CONTR_LIM, maxval=1+CONTR_LIM, seed=(idx, idx+1))
#     return aug_factor_or_zero(idx, data, factor)


# def metadata_brightness_factor(idx: int, data: PreprocessResponse) -> float:
#     factor = stateless_random_uniform(shape=[], minval=-BRIGHT_LIM, maxval=BRIGHT_LIM, seed=(idx, idx+1))
#     return aug_factor_or_zero(idx, data, factor)

# def get_counts_of_instances_per_class(idx: int, data: PreprocessResponse, label_flag: str = 'all') -> int:
#     data = data.data
#     x = data['samples'][idx]
#     all_labels = SUPERCATEGORY_CLASSES + categories
#     vehicle_labels = ['car'] + SUPERCATEGORY_CLASSES
#     catIds = [data['cocofile'].getCatIds(catNms=label)[0] for label in all_labels]  # keep same labels order
#     annIds = data['cocofile'].getAnnIds(imgIds=x['id'], catIds=catIds)
#     anns_list = data['cocofile'].loadAnns(annIds)
#     if label_flag == 'all':
#         return len(anns_list)   # all instances within labels
#     cat_name_to_id = dict(zip(all_labels, catIds))  # map label name to its ID
#     cat_id_counts = {cat_id: 0 for cat_id in catIds}    # counts dictionary
#     for ann in anns_list:
#         cat_id_counts[ann['category_id']] += 1
#     if label_flag == 'vehicle':  # count super category vehicle
#         vehicle_ids = [cat_name_to_id[cat_name] for cat_name in vehicle_labels]
#         return np.sum([cat_id_counts[cat_id] for cat_id in vehicle_ids])
#     cat_id = cat_name_to_id[label_flag]
#     return cat_id_counts[cat_id]
#
#
# def metadata_total_instances_count(idx: int, data: PreprocessResponse) -> int:
#     return get_counts_of_instances_per_class(idx, data, label_flag='all')
#
#
# def metadata_person_instances_count(idx: int, data: PreprocessResponse) -> int:
#     return get_counts_of_instances_per_class(idx, data, label_flag='person')
#
#
# def metadata_car_instances_count(idx: int, data: PreprocessResponse) -> int:
#     return get_counts_of_instances_per_class(idx, data, label_flag='car')
#
#
# def metadata_bus_instances_count(idx: int, data: PreprocessResponse) -> int:
#     return get_counts_of_instances_per_class(idx, data, label_flag='bus')
#
#
# def metadata_truck_instances_count(idx: int, data: PreprocessResponse) -> int:
#     return get_counts_of_instances_per_class(idx, data, label_flag='truck')
#
#
# def metadata_train_instances_count(idx: int, data: PreprocessResponse) -> int:
#     return get_counts_of_instances_per_class(idx, data, label_flag='train')
#
#
# def metadata_vehicle_instances_count(idx: int, data: PreprocessResponse) -> int:
#     return get_counts_of_instances_per_class(idx, data, label_flag='vehicle')
#
#
# def metadata_person_category_avg_size(idx: int, data: PreprocessResponse) -> float:
#     percent_val = metadata_person_category_percent(idx, data)
#     instances_cnt = metadata_person_instances_count(idx, data)
#     return np.round(percent_val/instances_cnt, 3) if instances_cnt > 0 else 0
#
#
# def metadata_car_vehicle_category_avg_size(idx: int, data: PreprocessResponse) -> float:
#     percent_val = metadata_car_vehicle_category_percent(idx, data)
#     if SUPERCATEGORY_GROUNDTRUTH:
#         instances_cnt = metadata_vehicle_instances_count(idx, data)
#     else:
#         instances_cnt = metadata_car_instances_count(idx, data)
#     return np.round(percent_val/instances_cnt, 3) if instances_cnt > 0 else 0

#todo:
#-------------------------------loss------------------------
def unnormalize_image(image: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return image*IMAGE_STD + IMAGE_MEAN


jet = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

def loss_visualizer(image: npt.NDArray[np.float32], prediction: npt.NDArray[np.float32], gt: npt.NDArray[np.float32]):
    image = unnormalize_image(image)
    ls = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')
    ls_image = ls(gt,prediction).numpy()
    ls_image = ls_image.clip(0,np.percentile(ls_image,95))
    ls_image /= ls_image.max()
    heatmap = scalarMap.to_rgba(ls_image)[..., :-1]
    # overlayed_image = ((heatmap * 0.4 + image * 0.6).clip(0,1)*255).astype(np.uint8)
    overlayed_image = ((heatmap).clip(0,1)*255).astype(np.uint8)
    return LeapImage(overlayed_image)

# ---------------------------------------------------------binding------------------------------------------------------

#preprocess function
leap_binder.set_preprocess(load_cityscapes_data_leap)
#TODO: unlables data

#set input and gt
leap_binder.set_input(input_image, 'normalized_image')
leap_binder.set_ground_truth(ground_truth_bbox, 'bbox')

#set prediction
leap_binder.add_prediction('object detection',
                               ["x", "y", "w", "h", "obj"] +
                                [cl for cl in categories])

#set meata_data
leap_binder.set_metadata(metadata_filename, DatasetMetadataType.string, 'filename')
leap_binder.set_metadata(metadata_city, DatasetMetadataType.string, 'city')
leap_binder.set_metadata(metadata_idx, DatasetMetadataType.float, 'idx')
leap_binder.set_metadata(metadata_gps_heading, DatasetMetadataType.float, 'gps_heading')
leap_binder.set_metadata(metadata_gps_latitude, DatasetMetadataType.float, 'gps_latitude')
leap_binder.set_metadata(metadata_gps_longtitude, DatasetMetadataType.float, 'gps_longtitude')
leap_binder.set_metadata(metadata_outside_temperature, DatasetMetadataType.float, 'outside_temperature')
leap_binder.set_metadata(metadata_speed, DatasetMetadataType.float, 'speed')
leap_binder.set_metadata(metadata_yaw_rate, DatasetMetadataType.float, 'yaw_rate')
leap_binder.set_metadata(number_of_bb, DatasetMetadataType.int, 'bb_count')
leap_binder.set_metadata(avg_bb_aspect_ratio, DatasetMetadataType.float, 'avg_bb_aspect_ratio')
leap_binder.set_metadata(avg_bb_area_metadata, DatasetMetadataType.float, 'avg_bb_area')
#TODO: SEE IF NEEDED
for i in range(4):
    leap_binder.set_metadata(is_class_exist_gen(i), DatasetMetadataType.float, f'does_{i}_exist')
#TODO: MORE METADATA FROM AMAZON

#set visualizer
leap_binder.set_visualizer(gt_bb_decoder, 'bb_gt_decoder', LeapDataType.ImageWithBBox)
leap_binder.set_visualizer(bb_decoder, 'bb_decoder', LeapDataType.ImageWithBBox)

# set custom metrics
leap_binder.add_custom_metric(regression_metric, "Regression_metric")
leap_binder.add_custom_metric(classification_metric, "Classification_metric")
leap_binder.add_custom_metric(object_metric, "Objectness_metric")

#TODO: custome loss
#leap_binder.set_visualizer(loss_visualizer,'loss_visualizer', LeapDataType.Image)




if __name__ == '__main__':
    responses = load_cityscapes_data_leap()
    input_image = input_image(0, responses[0])
    bounding_boxes_gt = ground_truth_bbox(0, responses[0])
    file_name = metadata_filename(0, responses[0])
    city = metadata_city(0, responses[0])
    idx = metadata_idx(0, responses[0])
    gps_heading = metadata_gps_heading(0, responses[0])
    gps_latitude = metadata_gps_latitude(0, responses[0])
    gps_longtitude = metadata_gps_longtitude(0, responses[0])
    outside_temperature = metadata_outside_temperature(0, responses[0])
    speed = metadata_speed(0, responses[0])
    yaw_rate = metadata_yaw_rate(0, responses[0])
