import json
import os

from PIL import Image

import numpy as np

from project_config import IMAGE_SIZE, MAX_BB_PER_IMAGE, BACKGROUND_LABEL
from leap_binder import load_cityscapes_data_leap
from utils_all.gcs_utils import _download
from utils_all.general_utils import extract_bounding_boxes_from_instance_segmentation_polygons, \
    filter_out_unknown_classes_id, polygon_to_bbox
from utils_all.preprocessing import load_cityscapes_data

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








