import numpy as np

from utils_all.preprocessing import CLASSES, FEATURE_MAPS, BOX_SIZES, OVERLAP_THRESH, \
    BACKGROUND_LABEL, MODEL_FORMAT, image_size, CONF_THRESH, NMS_THRESH, STRIDES, OFFSET

from code_loader.helpers.detection.yolo.loss import YoloLoss
from code_loader.helpers.detection.yolo.grid import Grid
from code_loader.helpers.detection.yolo.decoder import Decoder

#TODO: check about heat_maps and BACKGROUND_LABEL
BOXES_GENERATOR = Grid(image_size=image_size, feature_maps=FEATURE_MAPS, box_sizes=BOX_SIZES,
                       strides=STRIDES, offset=OFFSET)
DEFAULT_BOXES = BOXES_GENERATOR.generate_anchors()

LOSS_FN = YoloLoss(num_classes=CLASSES, overlap_thresh=OVERLAP_THRESH,
                   features=FEATURE_MAPS, anchors=np.array(BOX_SIZES),
                   default_boxes=DEFAULT_BOXES, background_label=BACKGROUND_LABEL,
                   from_logits=False if MODEL_FORMAT == "inference" else True,
                   image_size=image_size,
                   yolo_match=True,
                   semantic_instance=False)

DECODER = Decoder(CLASSES,
                  background_label=BACKGROUND_LABEL,
                  top_k=50,
                  conf_thresh=CONF_THRESH,
                  nms_thresh=NMS_THRESH,
                  semantic_instance=False,
                  max_bb=50,
                  max_bb_per_layer=50)
