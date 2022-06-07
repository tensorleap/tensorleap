from pathlib import Path
from torchvision.models.segmentation import deeplabv3_resnet50
# from coco_pytorch.unet import Unet
from coco_pytorch.utils import save_to_onnx

NUM_CLASSES = 4
data_shape = (3, 128, 128)


def leap_save_model(target_file_path: Path):
    # Load your model
    # model = Unet(NUM_CLASSES)
    model = deeplabv3_resnet50(pretrained=True, pretrained_backbone=True)   # use pytorch pretrained net
    # Save it to the path supplied as an argument
    save_to_onnx(model=model, input_size=data_shape, path=target_file_path)
