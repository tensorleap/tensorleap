# Adapted from https://www.tensorflow.org/tutorials/images/segmentation
import numpy as np
import tensorflow as tf
import coco.pix2pix_upsample as pix2pix
from coco.adapted_mobilenet_v2 import MobileNetV2


def get_output_indices(outputs, inputs):
    last_indices = np.zeros_like(outputs)
    for i, output in enumerate(outputs):
        last_indices[i] = len(tf.keras.Model(inputs=inputs, outputs=output).layers[1:])-1
    return last_indices


def encode_image(inputs, down_stack_layers, output_indices):
    outputs = []
    for i, layer in enumerate(down_stack_layers):
        if i == 0:
            x = layer(inputs)
        else:
            x = layer(x)
        if i in output_indices:
            outputs.append(x)
    return outputs


def unet_model(output_channels:int):
    base_model = MobileNetV2(
        input_shape=[128, 128, 3],
        include_top=False
    )  #Download on first run

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',  # 64x64
        'block_3_expand_relu',  # 32x32
        'block_6_expand_relu',  # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',  # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
    down_stack.trainable = False
    print(1)
    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),  # 32x32 -> 64x64
    ]

    inputs = tf.keras.layers.Input(shape=[128, 128, 3])

    # Downsampling through the model
  #  encoded_image = encode_image(inputs, down_stack_layers, output_indices)
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
      filters=output_channels, kernel_size=3, strides=2,
      padding='same')  #64x64 -> 128x128
    x = last(x)
    softmax = tf.keras.layers.Softmax(axis=-1)
    x = softmax(x)
    return tf.keras.Model(inputs=inputs, outputs=x)
    # return down_stack

