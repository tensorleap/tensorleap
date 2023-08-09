import tensorflow as tf
import numpy as np
from os.path import exists
import urllib
from keras.losses import CategoricalCrossentropy

from leap_binder import preprocess_func_leap, input_encoder_leap, gt_encoder, metadata_dict

def check_custom_integration():
    responses = preprocess_func_leap()

    if not exists('resnet.h5'):
        print("Downloading resnet for inference")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/example-datasets-47ml982d/resnet_cifar10/resnet.h5", "resnet.h5")
    resnet = tf.keras.models.load_model("resnet.h5")

    for i in range(0, 20):
        concat = np.expand_dims(input_encoder_leap(i, responses[0]), axis=0)
        y_pred = resnet([concat])
        gt = np.expand_dims(gt_encoder(i, responses[0]), axis=0)
        y_true = tf.convert_to_tensor(gt)

        ls = CategoricalCrossentropy()(y_true, y_pred).numpy()
        metadata = metadata_dict(i, responses[0])



if __name__ == '__main__':
    check_custom_integration()



