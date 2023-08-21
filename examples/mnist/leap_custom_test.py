from leap_binder import *
import tensorflow as tf
import os
import numpy as np
import pandas as pd

from mnist.config import CONFIG


def plot_horizontal_bar(y):
    df = pd.DataFrame({'labels': CONFIG['LABELS'], 'val': y})
    ax = df.plot.barh(x='labels', y='val')

def check_custom_test():
    print("started custom tests")
    responses = preprocess_func_leap()
    train = responses[0]
    val = responses[1]
    responses_set = val
    idx = 0

    #model
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = ('model/model.h5')
    cnn = tf.keras.models.load_model(os.path.join(dir_path, model_path))

    # get input and gt
    image = input_encoder(idx, responses_set)
    gt = gt_encoder(idx, responses_set)

    concat = np.expand_dims(image, axis=0)
    y_pred = cnn([concat])
    gt_expend = np.expand_dims(gt, axis=0)
    y_true = tf.convert_to_tensor(gt_expend)

    # get visualizer
    plot_horizontal_bar(gt)
    plot_horizontal_bar(y_pred.numpy().reshape(10,))

    #get meatdata
    sample_index = metadata_sample_index(idx, responses_set)
    one_hot_digit = metadata_one_hot_digit(idx, responses_set)

    print("finish tests")

if __name__=='__main__':
    check_custom_test()