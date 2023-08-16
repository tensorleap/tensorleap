from typing import Union, List
from numpy._typing import NDArray
import numpy as np
import tensorflow as tf
import json

from code_loader.contract.responsedataclasses import BoundingBox
from code_loader.helpers.detection.utils import xyxy_to_xywh_format
from code_loader.helpers.detection.yolo.utils import reshape_output_list
from code_loader.contract.datasetclasses import PreprocessResponse

from cityscapes_od.data.preprocess import Cityscapes, CATEGORIES
from cityscapes_od.utils.gcs_utils import _download
from cityscapes_od.config import CONFIG
from cityscapes_od.utils.yolo_utils import DECODER, DEFAULT_BOXES


def filter_out_unknown_classes_id(objects: List[dict]) -> List[dict]:
    """
    Description: This function takes a list of dictionaries objects as input and filters out unknown class IDs from it.
    Input: objects (List[dict]): A list of dictionaries, each representing an object with 'label' and 'polygon' keys.
    Output: new_objects (List[dict]): A filtered list of dictionaries containing objects with valid class IDs.
    """
    new_objects = []
    for object in objects:
        class_label = object['label']
        class_id = Cityscapes.get_class_id(class_label)
        if class_id is not None:
            new_object = {}
            new_object['label'] = class_id
            new_object['polygon'] = object['polygon']
            new_objects.append(new_object)
        else:
            continue
    return new_objects

def normelized_polygon(image_height: int, image_width: int, ann: dict) ->dict:
    """
    Description: This function normalizes a polygon using the height and width of the original image and a
                 dictionary ann representing an annotation (with 'polygon' key containing a list of (x, y) coordinates).
    Input: image_height (int): Height of the original image in pixels.
           image_width (int): Width of the original image in pixels.
           ann (dict): A dictionary representing an annotation with 'polygon' key containing a list of (x, y) coordinates.
    Output: ann (dict): The updated dictionary representing the annotation with normalized polygon coordinates.
    """

    normalized_height, normalized_width = CONFIG['IMAGE_SIZE'][0], CONFIG['IMAGE_SIZE'][1]
    coords = ann['polygon']
    new_coords = []
    for x, y in coords:
        new_x = x * (normalized_width / image_width)
        new_y = y * (normalized_height / image_height)
        new_coords.append([new_x, new_y])
    ann['polygon'] = new_coords
    return ann

def extract_bounding_boxes_from_instance_segmentation_polygons(json_data: dict) -> np.ndarray:
    """
    This function extracts bounding boxes from instance segmentation polygons present in the given JSON data.
    :param json_data: (dict) A dictionary containing instance segmentation polygons and image size information.
    :return: bounding_boxes: (numpy.ndarray) An array of bounding boxes in the format [x, y, width, height, class_id].
    """
    objects = json_data['objects']
    objects = filter_out_unknown_classes_id(objects)
    bounding_boxes = np.zeros([CONFIG['MAX_BB_PER_IMAGE'], 5])
    max_anns = min(CONFIG['MAX_BB_PER_IMAGE'], len(objects))
    original_image_size = (json_data['imgHeight'], json_data['imgWidth'])
    for i in range(max_anns):
        ann = objects[i]
        ann = normelized_polygon(original_image_size[0], original_image_size[1], ann)
        bbox = polygon_to_bbox(ann['polygon'])
        bbox /= np.array((CONFIG['IMAGE_SIZE'][0], CONFIG['IMAGE_SIZE'][1], CONFIG['IMAGE_SIZE'][0], CONFIG['IMAGE_SIZE'][1]))
        bounding_boxes[i, :4] = bbox
        bounding_boxes[i, 4] = ann['label']
    bounding_boxes[max_anns:, 4] = CONFIG['BACKGROUND_LABEL']
    return bounding_boxes

def polygon_to_bbox(polygon: List[List]) ->List[float]:
    """
    Converts a polygon representation to a bounding box representation.
    Args:
        vertices: (list) List of vertices defining the polygon. The vertices should be in the form [x1, y1, x2, y2, ...].
    Returns:
        list: Bounding box representation of the polygon in the form [x, y, width, height].
    Note:
        - The input list of vertices should contain x and y coordinates in alternating order.
        - The function calculates the minimum and maximum values of the x and y coordinates to determine the bounding box.
        - The bounding box representation is returned as [x, y, width, height], where (x, y) represents the center point of the
          bounding box, and width and height denote the size of the bounding box.
    """

    min_x = min(x for x, y in polygon)
    min_y = min(y for x, y in polygon)
    max_x = max(x for x, y in polygon)
    max_y = max(y for x, y in polygon)

    # Bounding box representation: (x_center, y_center, width, height)
    bbox = [(min_x + max_x) / 2., (min_y + max_y) / 2., max_x - min_x, max_y - min_y]
    return bbox


def bb_array_to_object(bb_array: Union[NDArray[float], tf.Tensor], iscornercoded: bool = True, bg_label: int = 0,
                       is_gt=False) -> List[BoundingBox]:
    """
    Assumes a (X,Y,W,H) Format for the BB text
    bb_array is (CLASSES,TOP_K,PROPERTIES) WHERE PROPERTIES =(conf,xmin,ymin,xmax,ymax)
    """
    bb_list = []
    if not isinstance(bb_array, np.ndarray):
        bb_array = np.array(bb_array)
    # fig, ax = plt.subplots(figsize=(6, 9)
    if len(bb_array.shape) == 3:
        bb_array = bb_array.reshape(-1, bb_array.shape[-1])
    for i in range(bb_array.shape[0]):
        if bb_array[i][-1] != bg_label:
            if iscornercoded:
                x, y, w, h = xyxy_to_xywh_format(bb_array[i][1:5])
                # unormalize to image dimensions
            else:
                x, y = bb_array[i][0], bb_array[i][1]
                w, h = bb_array[i][2], bb_array[i][3]
            conf = 1 if is_gt else bb_array[i][0]
            curr_bb = BoundingBox(x=x, y=y, width=w, height=h, confidence=conf,
                                  label=CATEGORIES[int(bb_array[i][min(5, len(bb_array[i]) - 1)])])

            bb_list.append(curr_bb)
    return bb_list

def get_predict_bbox_list(data: tf.Tensor) ->List[BoundingBox]:
    """
    Description: This function takes a TensorFlow tensor data as input and returns a list of bounding boxes representing predicted annotations.
    Input: data (tf.Tensor): A TensorFlow tensor representing the output data.
    Output: bb_object (List[BoundingBox]): A list of bounding box objects representing the predicted annotations.
    """
    from_logits = True if CONFIG['MODEL_FORMAT'] != "inference" else False
    decoded = False if CONFIG['MODEL_FORMAT'] != "inference" else True
    class_list_reshaped, loc_list_reshaped = reshape_output_list(
        np.reshape(data, (1, *data.shape)), decoded=decoded, image_size=CONFIG['IMAGE_SIZE'])
    # add batch
    outputs = DECODER(loc_list_reshaped,
                      class_list_reshaped,
                      DEFAULT_BOXES,
                      from_logits=from_logits,
                      decoded=decoded,
                      )
    bb_object = bb_array_to_object(outputs[0], iscornercoded=True, bg_label=CONFIG['BACKGROUND_LABEL'])
    return bb_object

def get_json(idx: int, data: PreprocessResponse) -> dict:
    """
    Description: This function takes an integer index idx and a PreprocessResponse object data as input and returns a
                Python dictionary containing JSON data.
    Input: idx (int): Index of the sample.
    data (PreprocessResponse): An object of type PreprocessResponse containing data attributes.
    Output: json_data (dict): A Python dictionary representing the JSON data obtained from the file at the given index.
    """
    data = data.data
    cloud_path = data['gt_bbx_path'][idx]
    fpath = _download(cloud_path)
    with open(fpath, 'r') as file:
        json_data = json.load(file)
    return json_data

def get_polygon(json_data: dict) -> List[dict]:
    """
    Description: This function takes a Python dictionary json_data as input and returns a list of dictionaries
    representing polygons.
    Input: json_data (dict): A Python dictionary representing the JSON data containing annotation information.
    Output: polygons (List[dict]): A list of dictionaries, each representing a label polygon.
    """
    polygons = []
    objects = json_data['objects']
    objects = filter_out_unknown_classes_id(objects)
    max_anns = min(CONFIG['MAX_BB_PER_IMAGE'], len(objects))
    for i in range(max_anns):
        polygon = objects[i]
        polygons.append(polygon)
    return polygons

def instances_num(valid_bbs) -> float:
    return float(valid_bbs.shape[0])

def avg_bb_aspect_ratio(valid_bbs) -> float:
    assert ((valid_bbs[:, 3] > 0).all())
    aspect_ratios = valid_bbs[:, 2] / valid_bbs[:, 3]
    return aspect_ratios.mean()

def avg_bb_area_metadata(valid_bbs) -> float:
    areas = valid_bbs[:, 2] * valid_bbs[:, 3]
    return areas.mean()

def count_small_bbs(bboxes) -> float:
    areas = bboxes[..., 2] * bboxes[..., 3]
    return float(len(areas[areas < CONFIG['SMALL_BBS_TH']]))

def number_of_bb(bboxes) -> int:
    number_of_bb = np.count_nonzero(bboxes[..., -1] != CONFIG['BACKGROUND_LABEL'])
    return number_of_bb