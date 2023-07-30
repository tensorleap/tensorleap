from typing import List
import numpy as np

from utils_all.preprocessing import BACKGROUND_LABEL
from utils_all.general_utils import bb_array_to_object
from yolo_helpers.yolo_utils import DECODER, DEFAULT_BOXES

from code_loader.contract.responsedataclasses import BoundingBox
from code_loader.helpers.detection.yolo.utils import reshape_output_list
from code_loader.contract.visualizer_classes import LeapImageWithBBox

def gt_bb_decoder(image, bb_gt) -> LeapImageWithBBox:
    bb_object: List[BoundingBox] = bb_array_to_object(bb_gt, iscornercoded=False, bg_label=BACKGROUND_LABEL,
                                                      is_gt=True)
    return LeapImageWithBBox(data=(image * 255).astype(np.float32), bounding_boxes=bb_object)


def bb_decoder(image, predictions):
    """
    Overlays the BB predictions on the image
    """
    class_list_reshaped, loc_list_reshaped = reshape_output_list(np.reshape(predictions, (1, predictions.shape[0]))) # add batch
    outputs = DECODER(loc_list_reshaped,
                      class_list_reshaped,
                      DEFAULT_BOXES,
                      from_logits=True
                      )
    bb_object = bb_array_to_object(outputs[0], iscornercoded=True, bg_label=BACKGROUND_LABEL)
    return LeapImageWithBBox((image * 255).astype(np.float32), bb_object)




