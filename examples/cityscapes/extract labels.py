import json
import os
from copy import deepcopy
from typing import Union
from PIL import Image
import numpy as np

from project_config import IMAGE_SIZE
from utils_all.gcs_utils import _download
from utils_all.general_utils import extract_bounding_boxes_from_instance_segmentation_polygons, \
    filter_out_unknown_classes_id, polygon_to_bbox
from utils_all.preprocessing import load_cityscapes_data, CATEGORIES

from code_loader.helpers.detection.utils import xywh_to_xyxy_format

def extract_bounding_boxes_from_instance_segmentation_polygons(json_data):
    """
    This function extracts bounding boxes from instance segmentation polygons present in the given JSON data.
    :param json_data: (dict) A dictionary containing instance segmentation polygons and image size information.
    :return: bounding_boxes: (numpy.ndarray) An array of bounding boxes in the format [x, y, width, height, class_id].
    """
    objects = json_data['objects']
    objects = filter_out_unknown_classes_id(objects)
    bounding_boxes = []
    #bounding_boxes = np.zeros([MAX_BB_PER_IMAGE, 5])
    #max_anns = min(MAX_BB_PER_IMAGE, len(objects))
    max_anns = len(objects)
    image_size = (json_data['imgHeight'], json_data['imgWidth'])
    for i in range(max_anns):
        ann = objects[i]
        bbox, min_x, min_y, max_x, max_y = polygon_to_bbox(ann['polygon'])
        bbox /= np.array((image_size[0], image_size[1], image_size[0], image_size[1])) #TODOD: need to replace with origin size
        bbox = list(bbox)
        bbox.append(ann['label'])
    #     bounding_boxes[i, :4] = bbox
    #     bounding_boxes[i, 4] = ann['label']
    # bounding_boxes[max_anns:, 4] = BACKGROUND_LABEL
        bounding_boxes.append(bbox)
    return bounding_boxes

def get_json_file(file_path):
    #fpath = _download(file_path)
    fpath = '/Users/chenrothschild/Desktop/Cityscapes_gtFine_trainvaltest_gtFine_train_ulm_ulm_000084_000019_gtFine_polygons.json'
    with open(fpath, 'r') as file:
        json_data = json.load(file)
    return json_data

def normelized_polygon(image_height, image_width, polygons):

    normalized_height, normalized_width = IMAGE_SIZE[0], IMAGE_SIZE[1]
    for polygon in polygons:
        coords = polygon['polygon']
        new_coords = []
        for x, y in coords:
            new_x = x * (normalized_width / image_width)
            new_y = y * (normalized_height / image_height)
            new_coords.append([new_x, new_y])
        polygon['polygon'] = new_coords
    return polygons

def non_normalized_image(file_path):
    fpath = _download(str(file_path))
    img = np.array(Image.open(fpath).convert('RGB').resize(IMAGE_SIZE))/255.
    return img


def export_to_yolo_format(bounding_boxes, output_file):
    with open(output_file, 'w') as f:
        for box in bounding_boxes:
            class_index = int(box[4])
            x_center, y_center, width, height = box[0], box[1], box[2], box[3]
            f.write(f"{class_index} {x_center} {y_center} {width} {height}\n")

dir_path = '/Users/chenrothschild/repo/tensorleap/examples/cityscapes/data/labels'

def get_file_name(file_path):
    # Get the base filename without the directory path
    base_filename = os.path.basename(file_path)

    # Remove the extension to get the filename without the extension
    filename_without_extension = os.path.splitext(base_filename)[0]

    # Split the filename using underscores and take the first two parts
    result = "_".join(filename_without_extension.split("_")[:3])
    file_name = result + ".txt"

    return file_name

def check_one():
    set = 'train'
    save_path = os.path.join(dir_path, set)
    file_path = 'Cityscapes/gtFine_trainvaltest/gtFine/train/ulm/ulm_000084_000019_gtFine_polygons.json'
    print(f'start {file_path}')
    #img = non_normalized_image(images[j])
    json_data = get_json_file(file_path)
    image_size = (json_data['imgHeight'], json_data['imgWidth'])
    polygons = json_data['objects']
    json_data['objects'] = normelized_polygon(image_size[0], image_size[1], polygons)
    bounding_boxes = extract_bounding_boxes_from_instance_segmentation_polygons(json_data)
    file_name = get_file_name(file_path)
    output_file_path = os.path.join(save_path, file_name)
    export_to_yolo_format(bounding_boxes, output_file_path)
    print(f'end {file_path}')



if __name__ == "__main__":

    check_one()
    all_images, all_gt_images, all_gt_labels, all_gt_labels_for_bbx, all_file_names, all_metadata, all_cities = \
        load_cityscapes_data()

    for i, set in enumerate(['train', 'val', 'test']):
        print(f'set: {set}')
        save_path = os.path.join(dir_path, set)
        all_files = all_gt_labels_for_bbx[i]
        images = all_images[i]
        for j, file_path in enumerate(all_files):
            try:
                print(f'start {file_path}')
                #img = non_normalized_image(images[j])
                json_data = get_json_file(file_path)
                image_size = (json_data['imgHeight'], json_data['imgWidth'])
                polygons = json_data['objects']
                json_data['objects'] = normelized_polygon(image_size[0], image_size[1], polygons)
                bounding_boxes = extract_bounding_boxes_from_instance_segmentation_polygons(json_data)
                file_name = get_file_name(file_path)
                output_file_path = os.path.join(save_path, file_name)
                export_to_yolo_format(bounding_boxes, output_file_path)
                print(f'end {file_path}')
            except:
                print(f"!wrong {file_path}")

#--------------------more functions-----------------------------------------------


def calculate_iou_all_pairs(bboxes: np.ndarray, image_size: int) -> np.ndarray: #TODO: no use
    """
    This function calculates the Intersection over Union (IOU) for all pairs of bounding boxes.
    :param bboxes: (numpy.ndarray) An array of bounding boxes in the format [x, y, width, height, class_id].
    :param image_size: (int) Size of the image (assumed to be square).
    :return: iou (numpy.ndarray) An array containing the calculated IOU values for all pairs of bounding boxes.
    """
    # Reformat all bboxes to (x_min, y_min, x_max, y_max)
    bboxes = np.asarray([xywh_to_xyxy_format(bbox[:-1]) for bbox in bboxes]) * image_size
    num_bboxes = len(bboxes)
    # Calculate coordinates for all pairs
    x_min = np.maximum(bboxes[:, 0][:, np.newaxis], bboxes[:, 0])
    y_min = np.maximum(bboxes[:, 1][:, np.newaxis], bboxes[:, 1])
    x_max = np.minimum(bboxes[:, 2][:, np.newaxis], bboxes[:, 2])
    y_max = np.minimum(bboxes[:, 3][:, np.newaxis], bboxes[:, 3])

    # Calculate areas for all pairs
    area_a = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    area_b = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

    # Calculate intersection area for all pairs
    intersection_area = np.maximum(x_max - x_min, 0) * np.maximum(y_max - y_min, 0)

    # Calculate union area for all pairs
    union_area = area_a[:, np.newaxis] + area_b - intersection_area

    # Calculate IOU for all pairs
    iou = intersection_area / union_area
    iou = iou[np.triu_indices(num_bboxes, k=1)]
    return iou

 #TODO: no use
def count_obj_bbox_occlusions(img: np.ndarray, bboxes: np.ndarray, occlusion_threshold: float, calc_avg_flag: bool) -> \
        Union[float, int]:
    """
    This function counts the number of occlusions for a specific object class within the given image.
    :param img: (numpy.ndarray) The image array.
    :param bboxes: (numpy.ndarray) An array of bounding boxes in the format [x, y, width, height, class_id].
    :param occlusion_threshold: (float) IOU threshold for considering a bounding box as occluded.
    :param calc_avg_flag: (bool) If True, returns the average occlusion count per object of the specified class.
    :return: If calc_avg_flag is True, the function returns the average occlusion count per object as a float.
             If calc_avg_flag is False, the function returns the total occlusion count for the specified object class as an integer.
    """
    img_size = img.shape[0]
    label = CATEGORIES.index('Object') #TODO: change
    obj_bbox = bboxes[bboxes[..., -1] == label]
    if len(obj_bbox) == 0:
        return 0.0
    else:
        ious = calculate_iou_all_pairs(obj_bbox, img_size)
        occlusion_count = len(ious[ious > occlusion_threshold])
        if calc_avg_flag:
            return int(occlusion_count / len(obj_bbox))
        else:
            return occlusion_count


def remove_label_from_bbs(bbs_object_array, removal_label, add_to_label): #TODO: no use
    """
    This function removes bounding boxes with a specific label and adds a new label suffix to the remaining bounding boxes.
    :param bbs_object_array: An array of BoundingBox objects.
    :param removal_label: The label of the bounding boxes to be removed.
    :param add_to_label: The suffix to be added to the label of the remaining bounding boxes.
    :return: new_bb_arr: A new array of BoundingBox objects after removing and updating labels.

    """
    new_bb_arr = []
    for bb in bbs_object_array:
        if bb.label != removal_label:
            new_bb = deepcopy(bb)
            new_bb.label = new_bb.label + "_" + add_to_label
            new_bb_arr.append(new_bb)
    return new_bb_arr


def calculate_overlap(box1, box2): #TODO: no use
    """
    This function calculates the overlap area between two bounding boxes.
    :param box1: box1 (tuple or list): The first bounding box represented as (x, y, width, height).
    :param box2: box2 (tuple or list): The second bounding box represented as (x, y, width, height).
    :return: overlap_area (float): The area of overlap between the two bounding boxes.
    """
    # Extract coordinates of the bounding boxes
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the coordinates of the intersection rectangle
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = max(0, min(x1 + w1, x2 + w2) - x_intersection)
    h_intersection = max(0, min(y1 + h1, y2 + h2) - y_intersection)

    # Calculate the overlap area
    overlap_area = w_intersection * h_intersection

    return overlap_area











