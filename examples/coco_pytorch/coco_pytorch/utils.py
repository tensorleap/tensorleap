import os
from typing import Tuple, Union
import torch


SAVE_PATH = os.path.join(os.getcwd())


# Function to Convert to ONNX
def save_to_onnx(model: torch.nn.Module, input_size: Tuple[int], model_name: Union[str, None] = None, path=None):
    # set the model to inference mode
    model.eval()
    dummy_input = torch.randn(1, *input_size, requires_grad=True)
    model_name = model._get_name() if model_name is None else model_name
    path = os.path.join(SAVE_PATH, f'{model_name}.onnx') if path is None else path
    torch.onnx.export(model,
                      dummy_input,
                      path,
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,
                      input_names=['modelInput'],
                      output_names=['modelOutput'],
                      dynamic_axes={'modelInput': {0: 'batch_size'},
                                    'modelOutput': {0: 'batch_size'}})

    print('Model has been converted to ONNX')
