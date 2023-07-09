import numpy as np
import pandas as pd

import tensorflow as tf
from datasets import load_dataset, concatenate_datasets

from enum import Enum
import nltk

nltk.download('punkt')

np.random.seed(0)
max_sequence_length = 384  # The maximum length of a feature (question and context)
max_answer_length = 20
to_squeeze = True
LABELS = ["start_idx", "end_idx"]
PAD_TOKEN = ''

# Preprocess Function
persistent_dir = '/nfs/'
TRAIN_SIZE, VAL_SIZE = 1000, 1000
# test_size = 500  # 1000 100
CHANGE_INDEX_FLAG = True
#tokenizer = AlbertTokenizerFast.from_pretrained("vumichien/albert-base-v2-squad2")


def load_data():
    np.random.seed(0)
    train_titles = ['New_York_City', 'American_Idol', 'Beyoncé']
    val_titles = ['Nikola_Tesla', 'Martin_Luther', 'Economic_inequality']
    #ds = load_dataset('squad') #if run locally
    ds = load_dataset('squad', cache_dir=persistent_dir)
    train_ds, val_ds = ds['train'], ds['validation']
    train_idx, val_idx = [], []
    for col in train_titles:
        idx = list(np.where(pd.Series(train_ds.data['title']) == col)[0])
        train_idx += idx
    for col in val_titles:
        idx = list(np.where(pd.Series(val_ds.data['title']) == col)[0])
        val_idx += idx
    train_idx = np.random.choice(train_idx, size=TRAIN_SIZE, replace=False) if TRAIN_SIZE < len(
        train_idx) else train_idx
    val_idx = np.random.choice(val_idx, size=VAL_SIZE, replace=False) if VAL_SIZE < len(val_idx) else val_idx
    train_idx.sort()
    val_idx.sort()

    enums_dic = {}
    for col in ["title"]:  # , "context"]:
        enum = Enum(col, train_titles + val_titles)
        enums_dic[col] = enum
    return train_idx, train_ds, val_idx, val_ds, enums_dic

#-----------------------------------------------------------------------------------------------

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
        end_position = 0
    else:
        # Otherwise it's the start and end token positions
        context_shifted_end = (offset[context_start:, 1] >= end_char).argmax()
        end_position = context_start + context_shifted_end
    return end_position




def get_readibility_score(analyzer_func):
    try:
        return float(np.round(analyzer_func().score, 3))

    except:
        return -1









