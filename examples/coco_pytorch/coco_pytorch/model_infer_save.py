from typing import Tuple
import torch
from utils import save_to_onnx
import onnx
from onnxsim import simplify




# run infer and save locally to test
def _infer_save_model(model: torch.nn.Module, input_size: Tuple[int, int, int], num_classes: int = 4):
    dummy_input = torch.randn(5, *input_size, requires_grad=True)
    out = model(dummy_input)
    save_to_onnx(model, input_size, model._get_name())


def _simplify_onnx(model_filename, out_filename=None):
    # load your predefined ONNX model
    model = onnx.load(model_filename)

    # convert model
    model_simp, check = simplify(model, dynamic_input_shape=True)

    assert check, "Simplified ONNX model could not be validated"

    if out_filename is not None:
        onnx.save(model_simp, out_filename)


