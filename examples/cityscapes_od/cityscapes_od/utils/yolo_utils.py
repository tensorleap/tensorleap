
import numpy as np
from cityscapes_od.config import CONFIG

from code_loader.helpers.detection.yolo.loss import YoloLoss
from code_loader.helpers.detection.yolo.grid import Grid
from code_loader.helpers.detection.yolo.decoder import Decoder

BOXES_GENERATOR = Grid(image_size=CONFIG['IMAGE_SIZE'], feature_maps=CONFIG['FEATURE_MAPS'], box_sizes=CONFIG['BOX_SIZES'],
                       strides=CONFIG['STRIDES'], offset=CONFIG['OFFSET'])
DEFAULT_BOXES = BOXES_GENERATOR.generate_anchors()

LOSS_FN = YoloLoss(num_classes=CONFIG['CLASSES'], overlap_thresh=CONFIG['OVERLAP_THRESH'],
                   features=CONFIG['FEATURE_MAPS'], anchors=np.array(CONFIG['BOX_SIZES']),
                   default_boxes=DEFAULT_BOXES, background_label=CONFIG['BACKGROUND_LABEL'],
                   from_logits=False if CONFIG['MODEL_FORMAT'] == "inference" else True,
                   image_size=CONFIG['IMAGE_SIZE'],
                   yolo_match=True,
                   semantic_instance=False)

DECODER = Decoder(CONFIG['CLASSES'],
                  background_label=CONFIG['BACKGROUND_LABEL'],
                  top_k=50,
                  conf_thresh=CONFIG['CONF_THRESH'],
                  nms_thresh=CONFIG['NMS_THRESH'],
                  semantic_instance=False,
                  max_bb=50,
                  max_bb_per_layer=50)
