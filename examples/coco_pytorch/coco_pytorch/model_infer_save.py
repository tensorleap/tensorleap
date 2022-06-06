from typing import Tuple
import torch
from utils import save_to_onnx

import imp
with open('.secret/db.py', 'rb') as fp:
    db = imp.load_module('.secret', fp, '.tensorleap/dataset.py', \
    ('.py', 'rb', imp.PY_SOURCE))



# run infer and save locally to test
def _infer_save_model(model: torch.nn.Module, input_size: Tuple[int, int, int], num_classes: int = 4):
    dummy_input = torch.randn(5, *input_size, requires_grad=True)
    out = model(dummy_input)
    save_to_onnx(model, input_size, model._get_name())







