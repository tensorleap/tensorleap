import tensorflow as tf
from albert.loss import CE_loss
from albert.metrices import CE_start_index, CE_end_index
from tensorleap import *
import numpy as np

def check():
    x = preprocess_load_article_titles()
    albert = tf.keras.models.load_model("/Users/chenrothschild/repo/tensorleap/examples/albert_QA/test/albert.h5")

    for idx in range(0, 20):
        input_keys = ['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping']
        for key in input_keys:
            concat = np.expand_dims(get_input_func(key)(idx, x[0]), axis=0)
        y_pred = albert([concat])
        gt = np.expand_dims(gt_index_encoder_leap(idx, x[0]), axis=0)
        y_true = tf.convert_to_tensor(gt)

        ce_ls = CE_loss(y_true, y_pred)
        exact_match_metric = exact_match_metric(y_true, y_pred)
        ce_start_index = CE_start_index(y_true, y_pred)
        ce_end_index = CE_end_index(y_true, y_pred)

        # ------- Metadata ---------
        answer_length = metadata_answer_length(idx, x[0])
        question_length = metadata_question_length(idx, x[0])
        context_length = metadata_context_length(idx, x[0])
        title = metadata_title(idx, x[0])
        title_idx = metadta_title_ids(idx, x[0])
        gt_string = metadata_gt_text(idx, x[0])
        is_truncated = metadata_is_truncated(idx, x[0])
        context_polarity = metadata_context_polarity(idx, x[0])
        context_subjectivity = metadata_context_subjectivity(idx, x[0])
        context_ari_score =  get_readibility_score(get_analyzer(idx, x[0]).ari)

        context_coleman_liau_score = get_readibility_score(get_analyzer(idx, x[0]).coleman_liau)
        context_dale_chall_score = get_readibility_score(get_analyzer(idx, x[0]).dale_chall)
        context_flesch_score = get_readibility_score(get_analyzer(idx, x[0]).flesch)
        context_flesch_kincaid_score = get_readibility_score(get_analyzer(idx, x[0]).flesch_kincaid)
        context_gunning_fog_score = get_readibility_score(get_analyzer(idx, x[0]).gunning_fog)
        context_linsear_write_score = get_readibility_score(get_analyzer(idx, x[0]).linsear_write)

        context_smog_score = get_readibility_score(get_analyzer(idx, x[0]).smog)
        context_spache_score = get_readibility_score(get_analyzer(idx, x[0]).spache)

        # Statistics metadata
        for stat in ['num_letters', 'num_words', 'num_sentences', 'num_polysyllabic_words', 'avg_words_per_sentence',
                     'avg_syllables_per_word']:
            state = get_statistics(stat, id, x[0], 'context')


if __name__=='__main__':
    check()