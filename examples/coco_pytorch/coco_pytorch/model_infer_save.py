import torch

from utils import save_to_onnx
from unet import Unet


def infer_save_model_test():
    model = Unet(4)
    input_size = (3, 128, 128)
    dummy_input = torch.randn(1, *input_size, requires_grad=True)
    out = model(dummy_input)
    save_to_onnx(model, (3, 128, 128), 'unet_scratch')


