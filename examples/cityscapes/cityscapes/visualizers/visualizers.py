import numpy as np
from code_loader.contract.visualizer_classes import LeapImageWithBBox

from cityscapes.utils.general_utils import get_mask_list


# Visualizers
def bb_decoder(image, bb_prediction):
    """
    Overlays the BB predictions on the image
    """
    bb_object, _ = get_mask_list(bb_prediction, None, is_gt=False)
    return LeapImageWithBBox((image * 255).astype(np.float32), bb_object)


def gt_bb_decoder(image, bb_gt) -> LeapImageWithBBox:
    bb_object, _ = get_mask_list(bb_gt, None, is_gt=True)
    return LeapImageWithBBox((image * 255).astype(np.float32), bb_object)





