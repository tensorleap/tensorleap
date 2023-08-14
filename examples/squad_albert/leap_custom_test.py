from os.path import exists
import urllib
import tensorflow as tf
import numpy as np

from leap_binder import preprocess_load_article_titles, get_input_func, gt_index_encoder_leap, metrics_dict, \
    metadata_dict
from squad_albert.loss import CE_loss


def check():
    x = preprocess_load_article_titles()

    if not exists('albert.h5'):
        print("Downloading resnet for inference")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/example-datasets-47ml982d/albert_squad/albert.h5", "albert.h5")
    albert = tf.keras.models.load_model("albert.h5")

    for idx in range(0, 20):
        input_keys = ['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping']
        inputs = []
        for key in input_keys:
            concat = np.expand_dims(get_input_func(key)(idx, x[0]), axis=0)
            inputs.append(concat)
        y_pred = albert([inputs])
        gt = np.expand_dims(gt_index_encoder_leap(idx, x[0]), axis=0)
        y_true = tf.convert_to_tensor(gt)

        ce_ls = CE_loss(y_true, y_pred)
        metrics_all = metrics_dict(y_true, y_pred)

        # ------- Metadata ---------
        meat_data_all = metadata_dict(idx, x[0])


if __name__=='__main__':
    check()