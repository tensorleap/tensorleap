from tensorleap import *
import numpy as np

def check():
    x = preprocess_load_article_titles()
    albert = tf.keras.models.load_model("/Users/chenrothschild/repo/tensorleap/examples/albert_QA/test/albert.h5")

    for i in range(0, 20):
        input_keys = ['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping']
        for key in input_keys:
            concat = np.expand_dims(get_input_func(key)(i, x[0]), axis=0)
        y_pred = albert([concat])
        gt = np.expand_dims(gt_index_encoder_leap(i, x[0]), axis=0)
        y_true = tf.convert_to_tensor(gt)

        ce_ls = CE_loss(y_true, y_pred)
        exact_match_metric = exact_match_metric(y_true, y_pred)
        ce_start_index = CE_start_index(y_true, y_pred)
        ce_end_index = CE_end_index(y_true, y_pred)

        # ------- Metadata ---------
        answer_length = metadata_answer_length(i, x[0])
        question_length = metadata_question_length(i, x[0])
        context_length = metadata_context_length(i, x[0])
        title = metadata_title(i, x[0])
        title_idx = metadta_title_ids(i, x[0])
        gt_string = metadata_gt_text(i, x[0])
        is_truncated = metadata_is_truncated(i, x[0])
        context_polarity = metadata_context_polarity(i, x[0])
        context_subjectivity = metadata_context_subjectivity(i, x[0])
        context_ari_score = lambda i, preprocess: get_readibility_score(get_analyzer(i, x[0]).ari)

        context_coleman_liau_score = lambda idx, preprocess: get_readibility_score(get_analyzer(i, x[0]).coleman_liau)
        context_dale_chall_score = lambda idx, preprocess: get_readibility_score(get_analyzer(i, x[0]).dale_chall)
        context_flesch_score = lambda idx, preprocess: get_readibility_score(get_analyzer(i, x[0]).flesch)
        context_flesch_kincaid_score = lambda idx, preprocess: get_readibility_score(get_analyzer(i, x[0]).flesch_kincaid)
        context_gunning_fog_score = lambda idx, preprocess: get_readibility_score(get_analyzer(i, x[0]).gunning_fog)
        context_linsear_write_score = lambda idx, preprocess: get_readibility_score(get_analyzer(i, x[0]).linsear_write)

        context_smog_score = lambda idx, preprocess: get_readibility_score(get_analyzer(i, x[0]).smog)
        context_spache_score = lambda idx, preprocess: get_readibility_score(get_analyzer(i, x[0]).spache)

        # Statistics metadata
        for stat in ['num_letters', 'num_words', 'num_sentences', 'num_polysyllabic_words', 'avg_words_per_sentence',
                     'avg_syllables_per_word']:
            state = lambda idx, preprocess, key=stat: get_statistics(key, i, x[0], 'context')


if __name__=='__main__':
    check()