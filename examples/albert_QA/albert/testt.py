from typing import List, Union, Tuple
from functools import lru_cache
import os
import pickle
import numpy as np
import pandas as pd
import numpy.typing as npt
import tensorflow as tf
from transformers import AlbertTokenizerFast, squad_convert_examples_to_features
from datasets import load_dataset, concatenate_datasets

# Tensorleap imports
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.enums import Metric, DatasetMetadataType, LeapDataType
from code_loader.contract.visualizer_classes import LeapText, LeapTextMask
from enum import Enum
from textblob import TextBlob
#from readability import Readability
import nltk

nltk.download('punkt')

#version20
np.random.seed(0)
max_sequence_length = 384  # The maximum length of a feature (question and context)
max_answer_length = 20
# doc_stride = 128  # The authorized overlap between two part of the context when splitting it is needed.
# doc_stride, max_query_length = 7, 6  # 128, 64

to_squeeze = True

LABELS = ["start_idx", "end_idx"]

PAD_TOKEN = ''

# Preprocess Function
#persistent_dir = '/nfs/'
persistent_dir = 'examples/albert_QA/data'
TRAIN_SIZE, VAL_SIZE = 1000, 1000
# test_size = 500  # 1000 100
CHANGE_INDEX_FLAG = True

def load_data():
    np.random.seed(0)
    train_ds, test_ds = load_dataset("squad", cache_dir=persistent_dir, split=[f'train[0:{TRAIN_SIZE}]', f'train[{TRAIN_SIZE}:{TRAIN_SIZE + VAL_SIZE}]'])
    return train_ds, test_ds

def preprocess_load_article_titles() -> List[PreprocessResponse]:
    np.random.seed(0)
    # train_titles = ['New_York_City', 'American_Idol', 'Beyoncé'] #, 'Queen_Victoria', 'Queen_Victoria']
    # val_titles = ['Nikola_Tesla', 'Martin_Luther', 'Economic_inequality'] #, 'Super_Bowl_50', 'Doctor_Who', 'American_Broadcasting_Company']

    ds = load_dataset('squad', cache_dir=persistent_dir)
    train_ds, val_ds = ds['train'], ds['validation']
    train_titles = pd.unique(train_ds.data['title']).tolist()
    val_titles = pd.unique(val_ds.data['title']).tolist()
    train_idx, val_idx = [], []
    for col in train_titles:
        idx = list(np.where(pd.Series(train_ds.data['title']) == col)[0])
        train_idx += idx
    for col in val_titles:
        idx = list(np.where(pd.Series(val_ds.data['title']) == col)[0])
        val_idx += idx
    train_idx = np.random.choice(train_idx, size=len(train_ds), replace=False) if len(train_ds) < len(train_idx) else train_idx
    val_idx = np.random.choice(val_idx, size=len(val_ds), replace=False) if len(val_ds) < len(val_idx) else val_idx
    train_idx.sort()
    val_idx.sort()

    enums_dic = {}
    for col in ["title"]:
        enum = Enum(col, train_titles+val_titles)
        enums_dic[col] = enum
    train = PreprocessResponse(length=len(train_idx), data={'ds': train_ds, 'idx': train_idx, **enums_dic})
    test = PreprocessResponse(length=len(val_idx), data={'ds': val_ds, 'idx': val_idx, **enums_dic})
    tokenizer = AlbertTokenizerFast.from_pretrained("vumichien/albert-base-v2-squad2")
    leap_binder.cache_container["tokenizer"] = tokenizer
    return [train, test]

[train, test] = preprocess_load_article_titles()
def load_sample_data():
    np.random.seed(0)
    train_max_size = 87500
    test_max_size = 10570
    batch = 1000
    jump = batch#*10
    assert TRAIN_SIZE >= batch and VAL_SIZE >= batch
    length_train_idx = int(TRAIN_SIZE // batch)
    length_val_idx = int(VAL_SIZE // batch)
    length_test_idx = int(VAL_SIZE // batch)
    assert jump > batch - 1 and train_max_size // jump >= length_train_idx + length_val_idx and test_max_size // jump >= length_test_idx
    train_sampled_idx = np.random.choice(np.arange(0, train_max_size, jump), size=length_train_idx + length_val_idx, replace=False)
    test_sampled_idx = np.random.choice(np.arange(0, test_max_size, jump), size=length_test_idx, replace=False)
    train_ds = load_dataset('squad', split=[f'train[{k}:{k + batch}]' for k in train_sampled_idx[:length_train_idx]], cache_dir=persistent_dir)
    val_ds = load_dataset('squad', split=[f'train[{k}:{k + batch}]' for k in train_sampled_idx[length_train_idx: length_train_idx + min(length_val_idx, len(train_sampled_idx))]], cache_dir=persistent_dir)
    test_ds = load_dataset('squad', split=[f'validation[{k}:{k + batch}]' for k in test_sampled_idx], cache_dir=persistent_dir)
    train_ds = concatenate_datasets(train_ds)
    val_ds = concatenate_datasets(val_ds)
    test_ds = concatenate_datasets(test_ds)
    return train_ds, val_ds, test_ds


@lru_cache
def preprocess_data() -> List[PreprocessResponse]:
    train, val, test = load_sample_data()
    enums_dic = {}
    for col in ["title"]:  # , "context"]:
        enum = Enum(col, np.unique(train[col] + val[col] + test[col]).tolist())
        enums_dic[col] = enum
    train = PreprocessResponse(length=len(train), data={**{'ds': train}, **enums_dic})
    # val = PreprocessResponse(length=len(val), data={**{'ds': val}, **enums_dic})
    test = PreprocessResponse(length=len(test), data={**{'ds': test}, **enums_dic})
    tokenizer = AlbertTokenizerFast.from_pretrained("vumichien/albert-base-v2-squad2")
    leap_binder.cache_container["tokenizer"] = tokenizer
    return [train, test]


def get_tokenizer():
    return leap_binder.cache_container["tokenizer"]

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
        if to_squeeze:
            x = x.squeeze()
            return x[:max_sequence_length]
        return x[:, :max_sequence_length]

    input_func.__name__ = f"{key}"
    return input_func


def get_context_positions(token_type_ids):
    context_start = tf.argmax(token_type_ids)
    context_end = max_sequence_length - tf.argmax(token_type_ids[::-1]) - 1
    return int(context_start), int(context_end)


def get_start_position(sample, inputs):
    answer = sample["answers"]
    offset = inputs["offset_mapping"].numpy().squeeze()
    start_char = answer["answer_start"][0]
    end_char = answer["answer_start"][0] + len(answer["text"][0])
    # Find the start and end of the context
    sequence_ids = inputs["token_type_ids"].numpy().squeeze()
    context_start, context_end = get_context_positions(sequence_ids)

    # If the answer is not fully inside the context, label is (0, 0)
    if offset[context_start][0] > start_char or offset[context_end - 1][1] < end_char:
        start_position = 0
    else:
        context_shifted_start = (offset[context_start:, 0] > start_char).argmax() - 1
        start_position = context_start + context_shifted_start
    return start_position


def get_end_position(sample, inputs):
    answer = sample["answers"]
    offset = inputs["offset_mapping"].numpy().squeeze()
    start_char = answer["answer_start"][0]
    end_char = answer["answer_start"][0] + len(answer["text"][0])
    # Find the start and end of the context
    sequence_ids = inputs["token_type_ids"].numpy().squeeze()
    context_start, context_end = get_context_positions(sequence_ids)

    # If the answer is not fully inside the context, label is (0, 0)
    if offset[context_start][0] > start_char or offset[context_end - 1][1] < end_char:
        end_position = 0  # TODO: maybe convert to -1
    else:
        # Otherwise it's the start and end token positions
        context_shifted_end = (offset[context_start:, 1] >= end_char).argmax()
        end_position = context_start + context_shifted_end
    return end_position


def gt_start_index_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    samples = preprocess.data['ds'][idx]
    inputs = get_inputs(idx, preprocess)
    start_position = get_start_position(samples, inputs)
    one_hot = np.zeros(max_sequence_length)
    if start_position < max_sequence_length:
        one_hot[start_position] = 1
    else:
        print("answer start position is out of max sequence length")
    return one_hot


def gt_end_index_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    sample = preprocess.data['ds'][idx]
    inputs = get_inputs(idx, preprocess)
    end_position = get_end_position(sample, inputs)
    one_hot = np.zeros(max_sequence_length)
    if end_position < max_sequence_length:
        one_hot[end_position] = 1
    else:
        print("answer start position is out of max sequence length")
    return one_hot


def gt_index_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    idx = convert_index(idx, preprocess)
    sample = preprocess.data['ds'][idx]
    inputs = get_inputs(idx, preprocess)
    start_position = get_start_position(sample, inputs)
    one_hot = np.zeros((max_sequence_length, 2))
    if start_position < max_sequence_length:
        one_hot[start_position, 0] = 1
    else:
        print("answer start position is out of max sequence length")
    end_position = get_end_position(sample, inputs)
    if end_position < max_sequence_length:
        one_hot[end_position, 1] = 1
    else:
        print("answer end position is out of max sequence length")
    return one_hot


def metadata_answer_length(idx: int, preprocess: PreprocessResponse) -> int:
    idx = convert_index(idx, preprocess)
    start_ind = np.argmax(gt_start_index_encoder(idx, preprocess))
    end_ind = np.argmax(gt_end_index_encoder(idx, preprocess))
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
    decoded = tokenizer.decode(input_ids)
    decoded = decoded.split(' ')
    return int(len(decoded) > max_sequence_length)


def metadata_context_polarity(idx: int, preprocess: PreprocessResponse) -> float:
    idx = convert_index(idx, preprocess)
    text = preprocess.data['ds'][idx]['context']
    blob = TextBlob(text)
    val = blob.polarity
    if val is None:
        val = -1
    return val


def metadata_context_subjectivity(idx: int, preprocess: PreprocessResponse) -> float:
    idx = convert_index(idx, preprocess)
    text = preprocess.data['ds'][idx]['context']
    blob = TextBlob(text)
    val = blob.subjectivity
    if val is None:
        val = -1
    return val


def get_analyzer(idx: int, preprocess: PreprocessResponse, section='context'):
    idx = convert_index(idx, preprocess)
    text: str = preprocess.data['ds'][idx][section]
    try:
        analyzer = Readability(text)
    except:
        analyzer = None
    return analyzer


def metadata_context_ari_score(idx: int, preprocess: PreprocessResponse) -> float:
    analyzer = get_analyzer(idx, preprocess, 'context')
    # if analyzer is not None:
    try:
        return np.round(analyzer.ari().score, 3)
    except:
        return 0


def metadata_context_flesch_kincaid_score(idx: int, preprocess: PreprocessResponse) -> float:
    analyzer = get_analyzer(idx, preprocess, 'context')
    # if analyzer is not None:
    try:
        return np.round(analyzer.flesch_kincaid().score, 3)
    except:
        return 0


def metadata_context_dale_chall_score(idx: int, preprocess: PreprocessResponse) -> float:
    analyzer = get_analyzer(idx, preprocess, 'context')
    # if analyzer is not None:
    try:
        return np.round(analyzer.dale_chall().score, 3)
    except:
        return 0


def get_decoded_tokens(input_ids):  # refactor
    input_ids = input_ids.astype(np.int32).tolist()
    tokenizer = get_tokenizer()
    decoded = tokenizer.convert_ids_to_tokens(input_ids)
    ind = decoded.index('<pad>') if '<pad>' in decoded else None
    decoded = decoded[:ind] if ind is not None else decoded  # truncate
    decoded = [token.replace(chr(9601), '') for token in decoded]
    return decoded


def tokens_decoder(input_ids):
    decoded = get_decoded_tokens(input_ids)
    if len(decoded) < max_sequence_length:  # pad
        decoded += (max_sequence_length - len(decoded)) * [PAD_TOKEN]
    elif len(decoded) > max_sequence_length:  # truncate
        decoded = decoded[:max_sequence_length]
    return LeapText(decoded)


def segmented_tokens_decoder(input_ids, token_type_ids, gt_logits, pred_logits):
    mask: npt.NDArray[np.uint8] = np.zeros(len(input_ids))
    labels_mapping = {'other': 0,
                      'question': 1,
                      'context': 2,
                      'gt_answer': 3,
                      'pred_answer': 4,
                      'overlap': 5
                      }
    labels: List[str] = list(labels_mapping.keys())
    gt_start_logits, gt_end_logits = get_start_end_arrays(gt_logits)
    pred_start_logits, pred_end_logits = get_start_end_arrays(pred_logits)
    gt_start_index = int(tf.math.argmax(gt_start_logits, axis=-1))
    gt_end_index = int(tf.math.argmax(gt_end_logits, axis=-1))
    pred_start_index = int(tf.math.argmax(pred_start_logits, axis=-1))
    pred_end_index = int(tf.math.argmax(pred_end_logits, axis=-1))
    context_start, context_end = get_context_positions(token_type_ids)
    mask[1:int(context_start - 1)] = labels_mapping['question']
    mask[int(context_start) - 1:] = labels_mapping['context']
    mask[gt_start_index:gt_end_index + 1] = labels_mapping['gt_answer']
    mask[pred_start_index:pred_end_index + 1] = labels_mapping['pred_answer']
    start_overlap = max(gt_start_index, pred_start_index)
    end_overlap = min(gt_end_index, pred_end_index)
    if start_overlap < end_overlap:
        mask[start_overlap: end_overlap + 1] = labels_mapping['overlap']
    text = get_decoded_tokens(input_ids)
    mask = mask[:len(text)]  # trancate if needed
    return LeapTextMask(mask.astype(np.uint8), text, labels)



def tokens_question_decoder(input_ids, token_type_ids):
    input_ids = input_ids.astype(np.int32).tolist()
    tokenizer = get_tokenizer()
    context_start, context_end = get_context_positions(token_type_ids)
    input_ids = input_ids[1:int(context_start - 1)]
    decoded = tokenizer.decode(input_ids, max_seq_length=max_sequence_length)
    decoded = decoded.split(' ')
    return LeapText(decoded)


def tokens_context_decoder(input_ids, token_type_ids):
    input_ids = input_ids.astype(np.int32).tolist()
    tokenizer = get_tokenizer()
    context_start, context_end = get_context_positions(token_type_ids)
    input_ids = input_ids[int(context_start): int(context_end + 1)]
    decoded = tokenizer.decode(input_ids, max_seq_length=max_sequence_length)
    decoded = decoded.split(' ')
    return LeapText(decoded)


def answer_decoder(logits, input_ids, token_type_ids, offset_mapping):
    start_logits, end_logits = get_start_end_arrays(logits)
    input_ids = input_ids.astype(np.int32).tolist()
    tokenizer = get_tokenizer()
    start_index = int(tf.math.argmax(start_logits, axis=-1))
    end_index = int(tf.math.argmax(end_logits, axis=-1))
    selected_answer_ids = input_ids[start_index:end_index + 1]
    answer = tokenizer.decode(selected_answer_ids)
    answer = answer.split(" ")
    return LeapText(answer)


def get_start_end_arrays(array):
    start_arr = array[..., 0]
    end_arr = array[..., 1]
    return start_arr, end_arr


def onehot_to_indices(one_hot):
    start_logits, end_logits = get_start_end_arrays(one_hot)
    start_ind = int(tf.math.argmax(start_logits, axis=-1))
    end_ind = int(tf.math.argmax(end_logits, axis=-1))
    return LeapText([start_ind, end_ind])

def get_readibility_score(analyzer_func):
    try:
        return float(np.round(analyzer_func().score, 3))

    except:
        return -1

def get_statistics(key: str, idx: int, subset: PreprocessResponse, section='context'):
    analyzer = get_analyzer(idx, subset, section)
    if analyzer is not None:
        return float(analyzer.statistics()[str(key)])
    else:
        return -1


def retrieve_best_prediction_decoder(start_logits, end_logits, input_ids, token_type_ids, offset_mapping):
    input_ids = input_ids.astype(np.int32).tolist()
    start_logits = tf.squeeze(start_logits)
    end_logits = tf.squeeze(end_logits)
    n_best = 20
    max_answer_length = 30
    answers = []
    tokenizer = get_tokenizer()
    context_start, context_end = get_context_positions(token_type_ids)
    context_token_ids = input_ids[int(context_start): int(context_end + 1)]
    context = tokenizer.decode(context_token_ids, max_seq_length=max_sequence_length)
    start_indexes = np.argsort(start_logits)[-1: -n_best - 1: -1].tolist()
    end_indexes = np.argsort(end_logits)[-1: -n_best - 1: -1].tolist()
    for start_index in start_indexes:
        for end_index in end_indexes:
            # Skip answers that are not fully in the context
            if all(offset_mapping[start_index] == 0) or all(offset_mapping[end_index] == 0):
                continue
            # Skip answers with a length that is either < 0 or > max_answer_length
            if (
                    end_index < start_index
                    or end_index - start_index + 1 > max_answer_length
            ):
                continue

            answer = {
                "text": context[offset_mapping[start_index - 1][0] + 1: offset_mapping[end_index][1]],
                "logit_score": start_logits[start_index] + end_logits[end_index],
            }
            answers.append(answer)

    # Select the answer with the best score
    if len(answers) > 0:
        best_answer = max(answers, key=lambda x: x["logit_score"])
        predicted_answer = best_answer["text"].split(" ")
    else:
        predicted_answer = [""]
    return predicted_answer


def index_accuracy(gt_ind, pred_ind):
    if np.sum(gt_ind) > 0:
        pred_ind = np.argmax(pred_ind)
        gt_ind = np.argmax(gt_ind)
        return int(gt_ind == pred_ind)
    else:
        return 0



def CE_start_index(ground_truth: tf.Tensor, prediction: tf.Tensor) -> tf.Tensor:
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    start_gt, end_gt = get_start_end_arrays(ground_truth)
    start_pred, end_pred = get_start_end_arrays(prediction)
    return loss(start_gt, start_pred)

def CE_end_index(ground_truth: tf.Tensor, prediction: tf.Tensor) -> tf.Tensor:
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    start_gt, end_gt = get_start_end_arrays(ground_truth)
    start_pred, end_pred = get_start_end_arrays(prediction)
    return loss(end_gt, end_pred)

def CE_loss(ground_truth: tf.Tensor, prediction: tf.Tensor) -> tf.Tensor:
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    alpha = 1.0
    start_pred, end_pred = get_start_end_arrays(prediction)
    start_gt, end_gt = get_start_end_arrays(ground_truth)
    combined_loss = loss(start_gt, start_pred) + alpha * loss(end_gt, end_pred)
    return combined_loss

def exact_match_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:  # return batch
    """
    This checks if the prediction is identical for the gt and prediction
    y_true :[B, max_sequence_length, 2]
    y_pred: [B, max_sequence_length, 2]
    """
    is_argmax_equal = tf.equal(tf.argmax(y_true, axis=1), tf.argmax(y_pred, axis=1))
    is_exact_match = tf.reduce_all(is_argmax_equal, axis=1)
    return tf.cast(is_exact_match, tf.float64)


def get_nonnegative_tensor(tensor: tf.Tensor) -> tf.Tensor:
    return tf.where(tensor > 0, tensor, tf.zeros_like(tensor))


def f1_metric(y_true: tf.Tensor, y_pred: tf.Tensor):  # return batch
    """
    Computes the F1 metric (2*precision*recall)/(precision + recall)
    """
    start_pred, end_pred = tf.transpose(tf.argmax(y_pred, axis=1))
    end_pred += 1  # fix end bondary encoding
    start_true, end_true = tf.transpose(tf.argmax(y_true, axis=1))
    end_true += 1  # fix end bondary encoding
    number_of_matches = tf.minimum(end_pred, end_true) - tf.maximum(start_pred, start_true)
    number_of_matches = get_nonnegative_tensor(number_of_matches)
    amount_predicted = get_nonnegative_tensor(end_pred - start_pred)
    amount_truth = get_nonnegative_tensor(end_true - start_true)
    precision = tf.where(amount_predicted > 0, number_of_matches / amount_predicted,
                         tf.zeros_like(amount_predicted, dtype=tf.float64))
    recall = tf.where(amount_truth > 0, number_of_matches / amount_truth, tf.zeros_like(amount_truth, dtype=tf.float64))
    f1 = tf.where(precision + recall > 0, 2 * precision * recall / (precision + recall),
                  tf.zeros_like(recall, dtype=tf.float64))
    return f1

# Dataset binding functions to bind the functions above to the `Dataset Instance`.

leap_binder.set_preprocess(function=preprocess_load_article_titles)

# ------- Inputs ---------
input_keys = ['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping']
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
leap_binder.set_ground_truth(function=gt_index_encoder, name='indices_gt')

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
    leap_binder.set_metadata(lambda idx, preprocess, key=stat: get_statistics(key, idx, preprocess, 'context'), metadata_type=DatasetMetadataType.float, name=f'context_{stat}')
    # leap_binder.set_metadata(lambda idx, preprocess, key=stat: get_statistics(key, idx, preprocess, 'answer'), metadata_type=DatasetMetadataType.float, name=f'question_{stat}')


# ------- Loss and Metrics ---------
# leap_binder.add_prediction('indices', LABELS, [], [exact_match_metric, f1_metric, CE_start_index, CE_end_index])  # , custom_metrics=[index_accuracy])
leap_binder.add_custom_loss(CE_loss, 'qa_cross_entropy')
leap_binder.add_custom_metric(exact_match_metric, "exact_match_metric")
leap_binder.add_custom_metric(f1_metric, "f1_metric")
leap_binder.add_custom_metric(CE_start_index, "CE_start_index")
leap_binder.add_custom_metric(CE_end_index, "CE_end_index")

# ------- Visualizers  ---------
leap_binder.set_visualizer(answer_decoder, 'new_answer_decoder', LeapDataType.Text)
leap_binder.set_visualizer(onehot_to_indices, 'prediction_indices', LeapDataType.Text)
leap_binder.set_visualizer(onehot_to_indices, 'gt_indices', LeapDataType.Text)
leap_binder.set_visualizer(tokens_decoder, 'tokens_decoder', LeapDataType.Text)
leap_binder.set_visualizer(tokens_question_decoder, 'tokens_question_decoder', LeapDataType.Text)
leap_binder.set_visualizer(tokens_context_decoder, 'tokens_context_decoder', LeapDataType.Text)
leap_binder.set_visualizer(segmented_tokens_decoder, 'segmented_tokens_decoder', LeapDataType.TextMask)

# leap_binder.set_visualizer(answer_span_decoder, 'answer_span_decoder', LeapDataType.TextMask)
# leap_binder.set_visualizer(retrieve_best_prediction_decoder, 'retrieve_best_prediction_decoder', LeapDataType.Text)
