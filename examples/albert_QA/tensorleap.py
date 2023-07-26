import yaml
import tensorflow as tf
import numpy as np
from readability import Readability

# Tensorleap imports
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.enums import DatasetMetadataType, LeapDataType
from code_loader.contract.visualizer_classes import LeapText, LeapTextMask

from transformers import AlbertTokenizerFast
from typing import List

from albert.utils import load_data, CHANGE_INDEX_FLAG, max_sequence_length, get_context_positions, \
    get_readibility_score
from albert.decoders import segmented_tokens_decoder, get_decoded_tokens, tokenizer_decoder, context_polarity, \
    context_subjectivity, answer_decoder, tokens_decoder, tokens_question_decoder, tokens_context_decoder
from albert.encoders import gt_index_encoder, gt_end_index_encoder, gt_start_index_encoder
from albert.loss import CE_loss
from albert.metrices import get_start_end_arrays, exact_match_metric, f1_metric, CE_start_index, CE_end_index
from albert.project_config import input_keys

# with open('/Users/chenrothschild/repo/tensorleap/examples/albert_QA/albert/project_config.yaml', 'r') as f:
#     config_data = yaml.safe_load(f)

# -------------------------load_data--------------------------------
def preprocess_load_article_titles() -> List[PreprocessResponse]:
    train_idx, train_ds, val_idx, val_ds, enums_dic = load_data()
    train = PreprocessResponse(length=len(train_idx), data={'ds': train_ds, 'idx': train_idx, **enums_dic})
    test = PreprocessResponse(length=len(val_idx), data={'ds': val_ds, 'idx': val_idx, **enums_dic})
    tokenizer = AlbertTokenizerFast.from_pretrained("vumichien/albert-base-v2-squad2")
    leap_binder.cache_container["tokenizer"] = tokenizer
    return [train, test]


# ------- Inputs ---------

def convert_index(idx: int, preprocess: PreprocessResponse) -> int:
    if CHANGE_INDEX_FLAG:
        return int(preprocess.data['idx'][idx])
    return idx


def get_inputs(idx: int, preprocess: PreprocessResponse) -> dict:
    x = preprocess.data['ds'][idx]
    tokenizer = get_tokenizer()
    inputs = tokenizer(
        x["question"],
        x["context"],
        return_tensors="tf",
        padding='max_length',
        max_length=max_sequence_length,
        return_offsets_mapping=True
    )
    return inputs.data


def get_input_func(key: str):
    def input_func(idx: int, preprocess: PreprocessResponse):
        idx = convert_index(idx, preprocess)
        x = get_inputs(idx, preprocess)[key].numpy()
        x = x.squeeze()
        return x[:max_sequence_length]
        return x[:, :max_sequence_length]

    input_func.__name__ = f"{key}"
    return input_func


# -------------------- gt  -------------------
def gt_index_encoder_leap(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    idx = convert_index(idx, preprocess)
    sample = preprocess.data['ds'][idx]
    inputs = get_inputs(idx, preprocess)
    one_hot = gt_index_encoder(sample, inputs)
    return one_hot


def get_tokenizer():  # V
    return leap_binder.cache_container["tokenizer"]


def gt_end_index_encoder_leap(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    sample = preprocess.data['ds'][idx]
    inputs = get_inputs(idx, preprocess)
    one_hot = gt_end_index_encoder(sample, inputs)
    return one_hot


def gt_start_index_encoder_leap(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    sample = preprocess.data['ds'][idx]
    inputs = get_inputs(idx, preprocess)
    one_hot = gt_start_index_encoder(sample, inputs)
    return one_hot


# ---------------------- meta_data  --------------------
def get_decoded_tokens_leap(input_ids: np.ndarray)->List[str]:
    tokenizer = get_tokenizer()
    decoded = get_decoded_tokens(input_ids, tokenizer)
    return decoded


def metadata_answer_length(idx: int, preprocess: PreprocessResponse) -> int:
    idx = convert_index(idx, preprocess)
    start_ind = np.argmax(gt_start_index_encoder_leap(idx, preprocess))
    end_ind = np.argmax(gt_end_index_encoder_leap(idx, preprocess))
    return int(end_ind - start_ind + 1)


def metadata_context_length(idx: int, preprocess: PreprocessResponse) -> int:
    token_type_ids = get_input_func("token_type_ids")(idx, preprocess)
    context_start, context_end = get_context_positions(token_type_ids)
    return int(context_end - context_start + 1)


def metadata_question_length(idx: int, preprocess: PreprocessResponse) -> int:
    token_type_ids = get_input_func("token_type_ids")(idx, preprocess)
    context_start, context_end = get_context_positions(token_type_ids)
    return int(context_start - 1)


def metadata_title(idx: int, preprocess: PreprocessResponse) -> str:
    idx = convert_index(idx, preprocess)
    return preprocess.data['ds'][idx]['title']


def metadta_title_ids(idx: int, preprocess: PreprocessResponse) -> int:
    idx = convert_index(idx, preprocess)
    return preprocess.data['title'][preprocess.data['ds'][idx]['title']].value


def metadta_context_ids(idx: int, preprocess: PreprocessResponse) -> int:
    idx = convert_index(idx, preprocess)
    return preprocess.data['context'][preprocess.data['ds'][idx]['context']].value


def metadata_gt_text(idx: int, preprocess: PreprocessResponse) -> str:
    idx = convert_index(idx, preprocess)
    sample = preprocess.data['ds'][idx]
    return sample['answers']['text'][0]


def metadata_is_truncated(idx: int, preprocess: PreprocessResponse) -> int:
    input_ids = get_input_func("input_ids")(idx, preprocess)
    tokenizer = get_tokenizer()
    decoded = tokenizer_decoder(tokenizer, input_ids)
    return int(len(decoded) > max_sequence_length)


def metadata_context_polarity(idx: int, preprocess: PreprocessResponse) -> float:
    idx = convert_index(idx, preprocess)
    text = preprocess.data['ds'][idx]['context']
    val = context_polarity(text)
    return val


def metadata_context_subjectivity(idx: int, preprocess: PreprocessResponse) -> float:
    idx = convert_index(idx, preprocess)
    text = preprocess.data['ds'][idx]['context']
    val = context_subjectivity(text)
    return val


def get_analyzer(idx: int, preprocess: PreprocessResponse, section='context') -> Readability:
    idx = convert_index(idx, preprocess)
    text: str = preprocess.data['ds'][idx][section]
    try:
        analyzer = Readability(text)
    except:
        analyzer = None
    return analyzer


def get_statistics(key: str, idx: int, subset: PreprocessResponse, section='context') -> float:
    analyzer = get_analyzer(idx, subset, section)
    if analyzer is not None:
        return float(analyzer.statistics()[str(key)])
    else:
        return -1


# ------- Visualizers  ---------
def answer_decoder_leap(logits: tf.Tensor, input_ids: np.ndarray, token_type_ids, offset_mapping) -> LeapText:
    tokenizer = get_tokenizer()
    answer = answer_decoder(logits, input_ids, tokenizer)
    return LeapText(answer)


def onehot_to_indices(one_hot: np.ndarray) -> LeapText:
    start_logits, end_logits = get_start_end_arrays(one_hot)
    start_ind = int(tf.math.argmax(start_logits, axis=-1))
    end_ind = int(tf.math.argmax(end_logits, axis=-1))
    return LeapText([start_ind, end_ind])


def tokens_decoder_leap(input_ids: np.ndarray) -> LeapText:  # V
    decoded = get_decoded_tokens_leap(input_ids)
    decoded = tokens_decoder(decoded)
    return LeapText(decoded)


def tokens_question_decoder_leap(input_ids: np.ndarray, token_type_ids: np.ndarray) -> LeapText:
    tokenizer = get_tokenizer()
    decoded = tokens_question_decoder(input_ids, token_type_ids, tokenizer)
    return LeapText(decoded)


def tokens_context_decoder_leap(input_ids: np.ndarray, token_type_ids: np.ndarray) -> LeapText:
    tokenizer = get_tokenizer()
    decoded = tokens_context_decoder(input_ids, token_type_ids, tokenizer)
    return LeapText(decoded)


def segmented_tokens_decoder_leap(input_ids: np.ndarray, token_type_ids: np.ndarray, gt_logits: np.ndarray, pred_logits: np.ndarray) -> LeapTextMask:
    mask, text, labels = segmented_tokens_decoder(input_ids, token_type_ids, gt_logits, pred_logits)
    return LeapTextMask(mask.astype(np.uint8), text, labels)


# Dataset binding functions to bind the functions above to the `Dataset Instance`.

leap_binder.set_preprocess(function=preprocess_load_article_titles)

# ------- Inputs ---------
"""
# input_ids: Indices of positions of each input sequence tokens in the position embeddings.
#     Selected in the range [0, config.max_position_embeddings - 1].
* token_type_ids: Segment token indices to indicate first and second portions of the inputs.
    Indices are selected in [0, 1]:
* attention_mask: Mask to avoid performing attention on padding token indices.
    Mask values selected in [0, 1]:
"""
for key in input_keys:
    leap_binder.set_input(function=get_input_func(key), name=f"{key}")

# ------- GT ---------
leap_binder.set_ground_truth(function=gt_index_encoder_leap, name='indices_gt')

# ------- Metadata ---------
leap_binder.set_metadata(function=metadata_answer_length, metadata_type=DatasetMetadataType.float, name='answer_length')
leap_binder.set_metadata(function=metadata_question_length, metadata_type=DatasetMetadataType.float,
                         name='question_length')
leap_binder.set_metadata(function=metadata_context_length, metadata_type=DatasetMetadataType.float,
                         name='context_length')
leap_binder.set_metadata(function=metadata_title, metadata_type=DatasetMetadataType.string, name='title')
leap_binder.set_metadata(function=metadta_title_ids, metadata_type=DatasetMetadataType.int, name='title_idx')
# leap_binder.set_metadata(function=metadta_context_ids, metadata_type=DatasetMetadataType.int, name='context_idx')
leap_binder.set_metadata(function=metadata_gt_text, metadata_type=DatasetMetadataType.string, name='gt_string')
leap_binder.set_metadata(function=metadata_is_truncated, metadata_type=DatasetMetadataType.int, name='is_truncated')
leap_binder.set_metadata(function=metadata_context_polarity, metadata_type=DatasetMetadataType.float,
                         name='context_polarity')
leap_binder.set_metadata(function=metadata_context_subjectivity, metadata_type=DatasetMetadataType.float,
                         name='context_subjectivity')

# Calculate Automated Readability Index (ARI).
leap_binder.set_metadata(lambda idx, preprocess: get_readibility_score(get_analyzer(idx, preprocess).ari),
                         metadata_type=DatasetMetadataType.float, name='context_ari_score')
# Calculate Coleman Liau Index
leap_binder.set_metadata(lambda idx, preprocess: get_readibility_score(get_analyzer(idx, preprocess).coleman_liau),
                         metadata_type=DatasetMetadataType.float, name='context_coleman_liau_score')
# Calculate Dale Chall
leap_binder.set_metadata(lambda idx, preprocess: get_readibility_score(get_analyzer(idx, preprocess).dale_chall),
                         metadata_type=DatasetMetadataType.float, name='context_dale_chall_score')
# Calculate Flesch Reading Ease score
leap_binder.set_metadata(lambda idx, preprocess: get_readibility_score(get_analyzer(idx, preprocess).flesch),
                         metadata_type=DatasetMetadataType.float, name='context_flesch_score')
# Calculate Flesch-Kincaid Grade Level
leap_binder.set_metadata(lambda idx, preprocess: get_readibility_score(get_analyzer(idx, preprocess).flesch_kincaid),
                         metadata_type=DatasetMetadataType.float, name='context_flesch_kincaid_score')
# Calculate Gunning Fog score
leap_binder.set_metadata(lambda idx, preprocess: get_readibility_score(get_analyzer(idx, preprocess).gunning_fog),
                         metadata_type=DatasetMetadataType.float, name='context_gunning_fog_score')
# Calculate Linsear Write
leap_binder.set_metadata(lambda idx, preprocess: get_readibility_score(get_analyzer(idx, preprocess).linsear_write),
                         metadata_type=DatasetMetadataType.float, name='context_linsear_write_score')
# SMOG Index. `all_sentences` indicates whether SMOG should use a sample of 30 sentences, as described in the original paper, or if it should use all sentences in the text
leap_binder.set_metadata(lambda idx, preprocess: get_readibility_score(get_analyzer(idx, preprocess).smog),
                         metadata_type=DatasetMetadataType.float, name='context_smog_score')
# Spache Index
leap_binder.set_metadata(lambda idx, preprocess: get_readibility_score(get_analyzer(idx, preprocess).spache),
                         metadata_type=DatasetMetadataType.float, name='context_spache_score')

# Statistics metadata
for stat in ['num_letters', 'num_words', 'num_sentences', 'num_polysyllabic_words', 'avg_words_per_sentence',
             'avg_syllables_per_word']:
    leap_binder.set_metadata(lambda idx, preprocess, key=stat: get_statistics(key, idx, preprocess, 'context'),
                             metadata_type=DatasetMetadataType.float, name=f'context_{stat}')

# ------- Loss and Metrics ---------
leap_binder.add_custom_loss(CE_loss, 'qa_cross_entropy')
leap_binder.add_custom_metric(exact_match_metric, "exact_match_metric")
leap_binder.add_custom_metric(f1_metric, "f1_metric")
leap_binder.add_custom_metric(CE_start_index, "CE_start_index")
leap_binder.add_custom_metric(CE_end_index, "CE_end_index")

# ------- Visualizers  ---------
leap_binder.set_visualizer(answer_decoder_leap, 'new_answer_decoder', LeapDataType.Text)
leap_binder.set_visualizer(onehot_to_indices, 'prediction_indices', LeapDataType.Text)
leap_binder.set_visualizer(onehot_to_indices, 'gt_indices', LeapDataType.Text)
leap_binder.set_visualizer(tokens_decoder_leap, 'tokens_decoder', LeapDataType.Text)
leap_binder.set_visualizer(tokens_question_decoder_leap, 'tokens_question_decoder', LeapDataType.Text)
leap_binder.set_visualizer(tokens_context_decoder_leap, 'tokens_context_decoder', LeapDataType.Text)
leap_binder.set_visualizer(segmented_tokens_decoder_leap, 'segmented_tokens_decoder', LeapDataType.TextMask)
