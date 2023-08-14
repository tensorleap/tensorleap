import os

import tensorflow as tf
import numpy as np
from os.path import exists
import urllib
from keras.losses import CategoricalCrossentropy

from leap_binder import preprocess_func_leap, input_encoder_leap, gt_encoder, metadata_dict

def check_custom_integration():
    model_path = ('examples/cifar10_resnet/cifar10_resnet/model')
    resnet = tf.keras.models.load_model(os.path.join(model_path, 'resnet.h5'))

    responses = preprocess_func_leap()

    for i in range(0, 20):
        concat = np.expand_dims(input_encoder_leap(i, responses[0]), axis=0)
        y_pred = resnet([concat])
        gt = np.expand_dims(gt_encoder(i, responses[0]), axis=0)
        y_true = tf.convert_to_tensor(gt)

        ls = CategoricalCrossentropy()(y_true, y_pred).numpy()
        metadata = metadata_dict(i, responses[0])



if __name__ == '__main__':
    check_custom_integration()



