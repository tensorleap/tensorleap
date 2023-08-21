
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
    model_path = ('model/imdb-dense.h5')
    imdb_dense = tf.keras.models.load_model(os.path.join(dir_path, model_path))

    # get input and gt
    input = input_tokens(idx, responses_set)
    gt = gt_sentiment(idx, responses_set)

    concat = np.expand_dims(input, axis=0)
    y_pred = imdb_dense([concat])
    gt = np.expand_dims(gt, axis=0)
    y_true = tf.convert_to_tensor(gt)

    # get visualizer
    #1
    tokenizer = leap_binder.custom_tokenizer
    texts = tokenizer.sequences_to_texts([input])
    print(texts[0].split(' '))

    #2
    if y_true == [1.0, 0.0]:
        text = 'positive'
    else:
        text = 'negative'
    print(text)


    #get meatdata
    gt_mdata = gt_metadata(idx, responses_set)
    all_raw_md = all_raw_metadata(idx, responses_set)

    print("finish tests")

if __name__=='__main__':
    check_custom_test()
