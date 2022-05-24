from pathlib import Path
from coco_pytorch.unet import Unet
from coco_pytorch.utils import save_to_onnx

NUM_CLASSES = 4
data_shape = (3, 128, 128)

def leap_save_model(target_file_path: Path):

    # Load your model
    model = Unet(NUM_CLASSES)
    # Save it to the path supplied as an argument
    save_to_onnx(model, data_shape, 'unet_local', target_file_path)


