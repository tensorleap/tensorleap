from leap_binder import *
import tensorflow as tf
import os
import numpy as np
import onnxruntime
from keras.losses import BinaryCrossentropy

def check_custom_test():
    print("started custom tests")
    responses = preprocess_func()
    train = responses[0]
    val = responses[1]
    responses_set = train
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = 'model/fabriceyhc-bert-base-uncased-imdb.onnx'

    for idx in range(20):
        input__id = input_ids(idx, responses_set)
        attention__mask = attention_masks(idx, responses_set)
        token_type__id = token_type_ids(idx, responses_set)

        # get input and gt
        gt = gt_sentiment(idx, responses_set)
        y_true = tf.convert_to_tensor(np.expand_dims(gt, axis=0))

        # model
        sess = onnxruntime.InferenceSession(os.path.join(dir_path, model_path))

        # get inputs
        input_name_1 = sess.get_inputs()[0].name
        input_name_2 = sess.get_inputs()[1].name
        input_name_3 = sess.get_inputs()[2].name
        label_name = sess.get_outputs()[-1].name

        y_pred = sess.run([label_name], {input_name_1: np.expand_dims(input__id, 0),
                                         input_name_2: np.expand_dims(attention__mask, 0),
                                         input_name_3: np.expand_dims(token_type__id, 0)})

        class_probabilities = np.exp(y_pred[0]) / np.sum(np.exp(y_pred[0]), axis=1, keepdims=True)

        # get loss
        ls = BinaryCrossentropy()(y_true, class_probabilities)

        # get meatdata
        gt_mdata = gt_metadata(idx, responses_set)
        all_raw_md = all_raw_metadata(idx, responses_set)

        # get visualizer
        tokenizer = leap_binder.custom_tokenizer
        data = input__id.astype(np.int64)
        text = tokenizer_decoder(tokenizer, data)
        tokens = text.split()
        input_list = pad_list(tokens)
        print(f'input text: {input_list}')

        # text_gt_visualizer_func
        ohe = {"pos": [0., 1.0], "neg": [1.0, 0.]}
        text = []
        if (y_true[0].numpy() == np.array(ohe["pos"])).all():
            text.append("pos")
        else:
            text.append("neg")
        print(f'gt: {text[0]}')

    print("finish tests")

def check_custom_test_dense():
    print("started custom tests")
    responses = preprocess_func()
    train = responses[0]
    val = responses[1]
    responses_set = train
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = 'model/exported-model.h5'

    for idx in range(20):
        # get input and gt

        input__tokens = input_tokens(idx, responses_set)
        concat = np.expand_dims(input__tokens, axis=0)

        gt = gt_sentiment(idx, responses_set)
        y_true = tf.convert_to_tensor(np.expand_dims(gt, axis=0))

        # model
        model = tf.keras.models.load_model(os.path.join(dir_path, model_path))
        y_pred = model([concat])

        # get loss
        ls = BinaryCrossentropy()(y_true, y_pred)

        # get meatdata
        gt_mdata = gt_metadata(idx, responses_set)
        all_raw_md = all_raw_metadata(idx, responses_set)

        # # get visualizer
        tokenizer = leap_binder.custom_tokenizer
        texts = tokenizer.sequences_to_texts([input__tokens])
        text_input = texts[0].split(' ')
        text_input = [text for text in text_input]
        padded_list = pad_list(text_input)
        print(f'input text: {padded_list}')

        labels_names = [CONFIG['LABELS_NAMES'][index] for index in range(y_pred.shape[-1])]

        # text_gt_visualizer_func
        # ohe = {"pos": [0., 1.0], "neg": [1.0, 0.]}
        ohe = {"pos": [1.0, 0.], "neg": [0., 1.0]}
        text = []
        if (y_true[0].numpy() == np.array(ohe["pos"])).all():
            text.append("pos")
        else:
            text.append("neg")
        print(f'gt: {text[0]}')

    print("finish tests")


if __name__ == '__main__':
    check_custom_test()

