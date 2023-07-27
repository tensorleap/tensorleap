from typing import List, Dict, Tuple
import yaml
import numpy as np
import pandas as pd
from enum import Enum
import nltk
import tensorflow as tf

from datasets import load_dataset

from albert.project_config import *

nltk.download('punkt')
np.random.seed(0)
#tokenizer = AlbertTokenizerFast.from_pretrained("vumichien/albert-base-v2-squad2")


def load_data() -> Tuple[np.ndarray, dict, np.ndarray, dict, Dict[str, Enum]]:
    """
    Description: Loads the SQuAD dataset and splits it into training and validation sets based on predefined titles.
    Returns:
    train_idx (np.ndarray): Indices of samples in the training set.
    train_ds (dict): Training dataset containing the samples.
    val_idx (np.ndarray): Indices of samples in the validation set.
    val_ds (dict): Validation dataset containing the samples.
    enums_dic (Dict[str, Enum]): A dictionary with an enumeration for the "title" field containing possible values from training and validation titles.
    """
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

def get_context_positions(token_type_ids: np.ndarray) -> List[int]:
    """
    Description: Determines the start and end positions of the context within the input token_type_ids.
    Parameters:
    token_type_ids (np.ndarray): Array of token type IDs for the input sequence.
    Returns:
    context_start (int): The start position of the context.
    context_end (int): The end position of the context.
    """
    context_start = tf.argmax(token_type_ids)
    context_end = max_sequence_length - tf.argmax(token_type_ids[::-1]) - 1
    return int(context_start), int(context_end)


def get_start_position(sample: dict, inputs: dict) -> int:
    """
    Description: Calculates the start position of the answer within the context for a given sample.
    Parameters:
    sample (dict): A dictionary containing the sample data, including the answer.
    inputs (dict): A dictionary containing inputs related to the sample, such as offset_mapping and token_type_ids.
    Returns:
    start_position (int): The start position of the answer within the context. Returns 0 if the answer is not fully inside the context.
    """
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


def get_end_position(sample: dict, inputs: dict) -> int:
    """
    Description: Calculates the end position of the answer within the context for a given sample.
    Parameters:
    sample (dict): A dictionary containing the sample data, including the answer.
    inputs (dict): A dictionary containing inputs related to the sample, such as offset_mapping and token_type_ids.
    Returns:
    end_position (int): The end position of the answer within the context. Returns 0 if the answer is not fully inside the context.
    """
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


def get_readibility_score(analyzer_func) -> float:
    """
    Description: Computes the readability score using the provided analyzer function.
    Parameters:
    analyzer_func (Callable): A function that analyzes the readability of a text and returns a score.
    Returns:
    readability_score (float): The computed readability score, rounded to three decimal places. Returns -1 if an exception occurs during computation.
    """
    try:
        return float(np.round(analyzer_func().score, 3))
    except:
        return -1









