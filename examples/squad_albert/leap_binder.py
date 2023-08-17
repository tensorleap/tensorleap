import tensorflow as tf
import numpy as np
from readability import Readability

# Tensorleap imports
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.enums import LeapDataType
from code_loader.contract.visualizer_classes import LeapText, LeapTextMask

from transformers import AlbertTokenizerFast
from typing import List, Dict, Union

from squad_albert.config import CONFIG
from squad_albert.data.preprocess import load_data
from squad_albert.decoders import get_decoded_tokens, tokenizer_decoder, context_polarity, context_subjectivity, \
    answer_decoder, tokens_decoder, tokens_question_decoder, tokens_context_decoder, segmented_tokens_decoder
from squad_albert.encoders import gt_index_encoder, gt_end_index_encoder, gt_start_index_encoder
from squad_albert.loss import CE_loss
from squad_albert.metrics import get_start_end_arrays, exact_match_metric, f1_metric, CE_start_index, CE_end_index
from squad_albert.utils.utils import get_context_positions, get_readibility_score


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
    if CONFIG['CHANGE_INDEX_FLAG']:
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
        max_length=CONFIG['max_sequence_length'],
        return_offsets_mapping=True
    )
    return inputs.data


def get_input_func(key: str):
    def input_func(idx: int, preprocess: PreprocessResponse):
        idx = convert_index(idx, preprocess)
        x = get_inputs(idx, preprocess)[key].numpy()
        x = x.squeeze()
        return x[:CONFIG['max_sequence_length']]
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

def metadata_length(idx: int, preprocess: PreprocessResponse) -> Dict[str, int]:
    token_type_ids = get_input_func("token_type_ids")(idx, preprocess)
    context_start, context_end = get_context_positions(token_type_ids)
    context_length = int(context_end - context_start + 1)
    question_length = int(context_start - 1)

    res = {
        'context_length': context_length,
        'question_length': question_length
    }

    return res

def metadata_dict(idx: int, data: PreprocessResponse) -> Dict[str, Union[float, int, str]]:
    idx = convert_index(idx, data)

    metadata_functions = {
        "answer_length": metadata_answer_length,
        "title": metadata_title,
        "title_idx": metadta_title_ids,
        "gt_text": metadata_gt_text,
        "context_polarity": metadata_context_polarity,
        "context_subjectivity": metadata_context_subjectivity
    }

    res = dict()
    for func_name, func in metadata_functions.items():
        res[func_name] = func(idx, data)
    return res

def get_decoded_tokens_leap(input_ids: np.ndarray)->List[str]:
    tokenizer = get_tokenizer()
    decoded = get_decoded_tokens(input_ids, tokenizer)
    return decoded


def metadata_answer_length(idx: int, preprocess: PreprocessResponse) -> int:
    start_ind = np.argmax(gt_start_index_encoder_leap(idx, preprocess))
    end_ind = np.argmax(gt_end_index_encoder_leap(idx, preprocess))
    return int(end_ind - start_ind + 1)


def metadata_title(idx: int, preprocess: PreprocessResponse) -> str:
    return preprocess.data['ds'][idx]['title']


def metadta_title_ids(idx: int, preprocess: PreprocessResponse) -> int:
    return preprocess.data['title'][preprocess.data['ds'][idx]['title']].value


def metadata_gt_text(idx: int, preprocess: PreprocessResponse) -> str:
    sample = preprocess.data['ds'][idx]
    return sample['answers']['text'][0]


def metadata_is_truncated(idx: int, preprocess: PreprocessResponse) -> int:
    input_ids = get_input_func("input_ids")(idx, preprocess)
    tokenizer = get_tokenizer()
    decoded = tokenizer_decoder(tokenizer, input_ids)
    return int(len(decoded) > CONFIG['max_sequence_length'])


def metadata_context_polarity(idx: int, preprocess: PreprocessResponse) -> float:
    text = preprocess.data['ds'][idx]['context']
    val = context_polarity(text)
    return val


def metadata_context_subjectivity(idx: int, preprocess: PreprocessResponse) -> float:
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


def tokens_decoder_leap(input_ids: np.ndarray) -> LeapText:
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
    tokenizer = get_tokenizer()
    mask, text, labels = segmented_tokens_decoder(input_ids, token_type_ids, gt_logits, pred_logits, tokenizer)
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
for key in CONFIG['input_keys']:
    leap_binder.set_input(function=get_input_func(key), name=f"{key}")

# ------- GT ---------
leap_binder.set_ground_truth(function=gt_index_encoder_leap, name='indices_gt')

# ------- Metadata ---------
leap_binder.set_metadata(function=metadata_dict, name='metadata_dict')
leap_binder.set_metadata(function=metadata_length, name='metadata_length')
leap_binder.set_metadata(function=metadata_is_truncated, name='is_truncated')


readability_scores = [
    ("ARI", "ari"),
    ("Coleman Liau", "coleman_liau"),
    ("Dale Chall", "dale_chall"),
    ("Flesch Reading Ease", "flesch"),
    ("Flesch-Kincaid Grade Level", "flesch_kincaid"),
    ("Gunning Fog", "gunning_fog"),
    ("Linsear Write", "linsear_write"),
    ("SMOG Index", "smog"),
    ("Spache Index", "spache")
]

for score_name, method_name in readability_scores:
    leap_binder.set_metadata(
        lambda idx, preprocess, method_name=method_name: get_readibility_score(get_analyzer(idx, preprocess).__getattribute__(method_name)),
        name=f"context_{method_name.lower()}_score"
    )

# Statistics metadata
for stat in ['num_letters', 'num_words', 'num_sentences', 'num_polysyllabic_words', 'avg_words_per_sentence',
             'avg_syllables_per_word']:
    leap_binder.set_metadata(lambda idx, preprocess, key=stat: get_statistics(key, idx, preprocess, 'context'),
                             name=f'context_{stat}')

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

