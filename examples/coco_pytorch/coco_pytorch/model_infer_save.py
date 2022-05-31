from typing import Tuple

import torch
from utils import save_to_onnx

# run infer and save locally to test
def infer_save_model(model: torch.nn.Module, input_size: Tuple[int, int, int], num_classes: int = 4):
    dummy_input = torch.randn(1, *input_size, requires_grad=True)
    out = model(dummy_input)
    save_to_onnx(model, input_size, model._get_name())


