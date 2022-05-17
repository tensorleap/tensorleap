import os
from typing import Tuple
import torch
import torchvision.models as models

# https://github.com/qubvel/segmentation_models.pytorch/blob/740dab561ccf54a9ae4bb5bda3b8b18df3790025/segmentation_models_pytorch/base/model.py#L5


from torch import nn, Tensor
from torch.nn import functional as F

class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):

        # if use_batchnorm == "inplace" and InPlaceABN is None:
        #     raise RuntimeError(
        #         "In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
        #         + "To install see: https://github.com/mapillary/inplace_abn"
        #     )

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            # bias=not(use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        # if use_batchnorm == "inplace":
        #     bn = InPlaceABN(out_channels, activation="leaky_relu", activation_param=0.0)
        #     relu = nn.Identity()

        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)
        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class Activation(nn.Module):
    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            self.activation = nn.Softmax(dim=1, **params)
        elif name == "softmax":
            self.activation = nn.Softmax(**params)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(**params)
        elif name == "tanh":
            self.activation = nn.Tanh()
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError(
                f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh/"
                f"argmax/argmax2d/clamp/None; got {name}"
            )

    def forward(self, x):
        return self.activation(x)


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class Attention(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)

class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)


    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = nn.Softmax(activation)
        super().__init__(conv2d, upsampling, activation)



class UnetSegModel(torch.nn.Module):
    def __init__(self, input_shape: int, n_classes: int) -> torch.nn.Module:
        super(UnetSegModel, self).__init__()
        self.encoder = models.mobilenet_v2(pretrained=True).features
        dummy_input = torch.unsqueeze(torch.zeros((3, input_shape, input_shape)), 0)
        out_channels = self.encoder(dummy_input).shape#[1]
        decoder_channels = [256, 128, 64, 32, 16]
        self.decoder = UnetDecoder(
            encoder_channels=out_channels,
            decoder_channels=decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=n_classes,
            activation=None,
            kernel_size=3,
        )

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        # self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        return self.segmentation_head(decoder_output)



# Function to Convert to ONNX
def Convert_ONNX(model: torch.nn.Module, input_size: Tuple[int], model_name: str):

    # set the model to inference mode
    model.eval()

    # Let's create a dummy input tensor
    dummy_input = torch.randn(1, *input_size, requires_grad=True)

    # Export the model
    torch.onnx.export(model,         # model being run
         dummy_input,       # model input (or a tuple for multiple inputs)
         os.path.join(f"{model_name}.onnx"),       # where to save the model
         export_params=True,  # store the trained parameter weights inside the model file
         opset_version=11,    # the ONNX version to export the model to
         do_constant_folding=True,  # whether to execute constant folding for optimization
         input_names = ['modelInput'],   # the model's input names
         output_names = ['modelOutput'], # the model's output names
         dynamic_axes={'modelInput': {0 : 'batch_size'},    # variable length axes
                                'modelOutput' : {0 : 'batch_size'}})

    print('Model has been converted to ONNX')


# def _test_Convert_ONNX():
#     onnx_model = onnx.load_model(file_path)
#     input_names = [_input.name for _input in onnx_model.graph.input]
#     k_model = onnx_to_keras(onnx_model, input_names=input_names,
#                             name_policy='attach_weights_name')
#
#     return convert_to_keras_model(k_model)



# model = LRASPPMobileNet()
# model = UnetGenerator(128, 128)
model = UnetSegModel(input_shape=128, n_classes=3)
dummy_input = torch.randn(1, *(3, 128, 128), requires_grad=True)
out = model(dummy_input)
Convert_ONNX(model, (3, 128, 128), 'unet_model')

