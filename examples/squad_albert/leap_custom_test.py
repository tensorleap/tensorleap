import os
import tensorflow as tf
import numpy as np

from leap_binder import preprocess_load_article_titles, get_input_func, gt_index_encoder_leap, metadata_is_truncated, \
    metadata_length, metadata_dict, get_statistics, get_analyzer
from squad_albert.loss import CE_loss
from squad_albert.metrics import exact_match_metric, f1_metric, CE_start_index, CE_end_index
from squad_albert.utils.utils import get_readibility_score



def check():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = ('squad_albert/model/albert.h5')
    albert = tf.keras.models.load_model(os.path.join(dir_path, model_path))

    x = preprocess_load_article_titles()
    for idx in range(0, 20):
        input_keys = ['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping']
        inputs = []
        for key in input_keys:
            concat = np.expand_dims(get_input_func(key)(idx, x[0]), axis=0)
            inputs.append(concat)
        y_pred = albert([inputs])
        gt = np.expand_dims(gt_index_encoder_leap(idx, x[0]), axis=0)
        y_true = tf.convert_to_tensor(gt)

        #metrices
        ce_ls = CE_loss(y_true, y_pred)
        match_metric = exact_match_metric(y_true, y_pred)
        f1 = f1_metric(y_true, y_pred)
        CE_start_in = CE_start_index(y_true, y_pred)
        CE_end_in = CE_end_index(y_true, y_pred)

        # ------- Metadata ---------
        doct = metadata_dict(idx, x[0])
        length = metadata_length(idx, x[0])
        is_truncated = metadata_is_truncated(idx, x[0])


        is_truncated = metadata_is_truncated(idx, x[0])
        length = metadata_length(idx, x[0])
        meat_data_all = metadata_dict(idx, x[0])
        for stat in ['num_letters', 'num_words', 'num_sentences', 'num_polysyllabic_words', 'avg_words_per_sentence',
                     'avg_syllables_per_word']:
            state = get_statistics(stat, idx, x[0], 'context')

        for score in ['ari', 'coleman_liau', 'dale_chall', 'flesch', 'flesch_kincaid',
                      'gunning_fog', 'linsear_write', 'smog', 'spache']:
            score = get_readibility_score(get_analyzer(idx, x[0]).__getattribute__(score))



if __name__=='__main__':
    check()