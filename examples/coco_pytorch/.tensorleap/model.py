from pathlib import Path
import tensorflow as tf
from coco_pytorch.unet import Unet
from coco_pytorch.utils import save_to_onnx

OUTPUT_CLASSES = 4

def leap_save_model(target_file_path: Path):
    # Load your model
    model = Unet()
    # Save it to the path supplied as an argument
    model.save(target_file_path)

