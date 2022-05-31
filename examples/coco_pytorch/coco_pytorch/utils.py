import os
from typing import Tuple
import torch


SAVE_PATH = os.path.join(os.getcwd(), 'models')


# Function to Convert to ONNX
def save_to_onnx(model: torch.nn.Module, input_size: Tuple[int], model_name: str, path=None):
    # set the model to inference mode
    model.eval()

    # Let's create a dummy input tensor
    dummy_input = torch.randn(1, *input_size, requires_grad=True)
    path = os.path.join(SAVE_PATH, f'{model_name}.onnx') if path is None else path  # where to save the model
    # Export the model
    torch.onnx.export(model,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      path,  # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['modelInput'],  # the model's input names
                      output_names=['modelOutput'],  # the model's output names
                      dynamic_axes={'modelInput': {0: 'batch_size'},  # variable length axes
                                    'modelOutput': {0: 'batch_size'}})

    print('Model has been converted to ONNX')
