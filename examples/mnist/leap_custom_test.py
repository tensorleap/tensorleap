from leap_binder import *
import tensorflow as tf
import os
import numpy as np

def check_custom_test():
    print("started custom tests")
    responses = preprocess_func()
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
    gt = np.expand_dims(gt, axis=0)
    y_true = tf.convert_to_tensor(gt)

    # get visualizer

    #get meatdata
    sample_index = metadata_sample_index(idx, responses_set)
    label = metadata_label(idx, responses_set)
    label_name = metadata_label_name(idx, responses_set)
    even_odd = metadata_even_odd(idx, responses_set)
    circle = metadata_circle(idx, responses_set)

if __name__=='__main__':
    check_custom_test()