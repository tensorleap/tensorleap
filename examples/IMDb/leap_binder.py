from typing import List, Optional, Callable, Tuple, Dict
import tensorflow as tf
import json, os, re, string
from os.path import basename, dirname, join

import pandas as pd
import numpy as np



from pandas.core.frame import DataFrame as DataFrameType

# Tensorleap imports
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.enums import DatasetMetadataType, LeapDataType, Metric
from code_loader.contract.visualizer_classes import LeapText

from IMDb.config import CONFIG
from IMDb.data.preprocess import download_load_assets
from IMDb.gcs_utils import _download
from IMDb.utils import prepare_input



# Preprocess Function
def preprocess_func() -> List[PreprocessResponse]:
    tokenizer, df = download_load_assets()
    train_label_size = int(0.9 * CONFIG['NUMBER_OF_SAMPLES'] / 2)
    val_label_size = int(0.1 * CONFIG['NUMBER_OF_SAMPLES'] / 2)
    df = df[df['subset'] == 'train']
    train_df = pd.concat([df[df['gt'] == 'pos'][:train_label_size], df[df['gt'] == 'neg'][:train_label_size]],
                         ignore_index=True)
    val_df = pd.concat([df[df['gt'] == 'pos'][train_label_size:train_label_size + val_label_size],
                        df[df['gt'] == 'neg'][train_label_size:train_label_size + val_label_size]], ignore_index=True)
    ohe = {"pos": [1.0, 0.], "neg": [0., 1.0]}

    # Generate a PreprocessResponse for each data slice, to later be read by the encoders.
    # The length of each data slice is provided, along with the data dictionary.
    # In this example we pass `images` and `labels` that later are encoded into the inputs and outputs
    train = PreprocessResponse(length=2 * train_label_size, data={"df": train_df, "tokenizer": tokenizer, "ohe": ohe})
    val = PreprocessResponse(length=2 * val_label_size, data={"df": val_df, "tokenizer": tokenizer, "ohe": ohe})
    response = [train, val]

    # Adding custom data to leap_binder for later usage within the visualizer function
    leap_binder.custom_tokenizer = tokenizer

    return response


# Input Encoder - fetches the text with the index `idx` from the `paths` array set in
# the PreprocessResponse's data. Returns a numpy array containing padded tokenized input
def input_func(idx, preprocess):
    comment_path = preprocess.data['df']['paths'][idx]
    local_path = _download(comment_path)
    with open(local_path, 'r') as f:
        comment = f.read()
    tokenizer = preprocess.data['tokenizer']
    padded_input = prepare_input(tokenizer, comment)
    return padded_input

def input_tokens(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    padded_input = input_func(idx, preprocess)
    padded_input = padded_input.squeeze()
    return padded_input

# Input Encoder - fetches the text with the index `idx` from the `paths` array set in
# the PreprocessResponse's data. Returns a numpy array containing padded tokenized input.
def input_ids(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    padded_input = input_func(idx, preprocess)
    padded_input = np.array(padded_input['input_ids'])
    padded_input = padded_input.squeeze()
    return padded_input

def attention_masks(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    padded_input = input_func(idx, preprocess)
    padded_input = np.array(padded_input['attention_mask'])
    padded_input = padded_input.squeeze()
    return np.array(padded_input)

def token_type_ids(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    padded_input = input_func(idx, preprocess)
    padded_input = np.array(padded_input['token_type_ids'])
    padded_input = padded_input.squeeze()
    return np.array(padded_input)

# Ground Truth Encoder - fetches the label with the index `idx` from the `gt` array set in
# the PreprocessResponse's  data. Returns a numpy array containing a hot vector label correlated with the sample.
def gt_sentiment(idx: int, preprocess: PreprocessResponse) -> List[float]:
    gt_str = preprocess.data['df']['gt'][idx]
    return preprocess.data['ohe'][gt_str]

def gt_metadata(idx: int, preprocess: PreprocessResponse) -> str:
    if preprocess.data['df']['gt'][idx] == "pos":
        return "positive"
    else:
        return "negative"

def all_raw_metadata(idx: int, preprocess: PreprocessResponse):
    df = preprocess.data['df']

    res = {
        'automated_readability_index_metadata': df['automated_readability_index'][idx],
        'coleman_liau_index_metadata': df['coleman_liau_index'][idx],
        'crawford_metadata': df['crawford'][idx],
        'dale_chall_readability_score_metadata': df['dale_chall_readability_score'][idx],
        'difficult_words_metadata': df['difficult_words'][idx],
        'fernandez_huerta_metadata': df['fernandez_huerta'][idx],
        'flesch_kincaid_metadata': df['flesch_kincaid'][idx],
        'flesch_reading_metadata': df['flesch_reading'][idx],
        'gulpease_index_metadata': df['gulpease_index'][idx],
        'gunning_fog_metadata': df['gunning_fog'][idx],
        'gutierrez_polini_metadata': df['gutierrez_polini'][idx],
        'length_metadata': df['length'][idx],
        'linsear_write_formula_metadata': df['linsear_write_formula'][idx],
        'oov_count_metadata': df['oov_count'][idx],
        'osman_metadata': df['osman'][idx],
        'polarity_metadata': df['polarity'][idx],
        'subjectivity_metadata': df['subjectivity'][idx],
        'szigriszt_pazos_metadata': df['szigriszt_pazos'][idx],
    }

    for key, value in res.items():
        if not isinstance(value, (float, np.float64)):
            res[key] = np.float64(value)
    return res

def tokenizer_decoder(tokenizer, input_ids: np.ndarray) -> List[str]:
    """
    Description: Decodes the input tokens from their corresponding input IDs using the provided tokenizer.
    Parameters:
    tokenizer: The tokenizer used to convert token IDs to text.
    input_ids (np.ndarray): Array of input token IDs.
    Returns:
    decoded (List[str]): List of decoded tokens as strings.
    """
    decoded = tokenizer.decode(input_ids)
    return decoded



def tokenizer_decoder_gt(tokenizer, gt: np.ndarray) -> str:
    """
    Description: Decodes the input tokens from their corresponding input IDs using the provided tokenizer.
    Parameters:
    tokenizer: The tokenizer used to convert token IDs to text.
    input_ids (np.ndarray): Array of input token IDs.
    Returns:
    decoded (List[str]): List of decoded tokens as strings.
    """
    if gt[0] == 1.0:
        return 'Positive'
    return 'Negative'

# Visualizer functions define how to interpet the data and visualize it.
# In this example we define a tokens-to-text visualizer.
def text_visualizer_func(input_ids: np.ndarray) -> LeapText:
    tokenizer = leap_binder.custom_tokenizer
    text = tokenizer_decoder(tokenizer, input_ids)
    tokens = [token for token in text.split('[PAD]') if token.strip() != '']
    return LeapText(tokens)

def text_visualizer_func_old(data: np.ndarray) -> LeapText:
    tokenizer = leap_binder.custom_tokenizer
    texts = tokenizer.sequences_to_texts([data])
    text_input = texts[0].split(' ')
    text_input = [text for text in text_input if text != '[OOV]']
    return LeapText(text_input)

def text_visualizer_output(y_true) -> LeapText:
    ohe = {"pos": [1.0, 0.], "neg": [0., 1.0]}
    text = []
    if (y_true[0] == np.array(ohe["pos"])).all():
        text.append("pos")
    else:
        text.append("neg")
    return LeapText(text)


# Binders
leap_binder.set_preprocess(function=preprocess_func)
# leap_binder.set_input(function=input_tokens, name='tokens')
leap_binder.set_input(function=input_ids, name='input_ids')
leap_binder.set_input(function=attention_masks, name='attention_masks')
leap_binder.set_input(function=token_type_ids, name='token_type_ids')
leap_binder.set_ground_truth(function=gt_sentiment, name='sentiment')
leap_binder.set_metadata(function=gt_metadata, name='gt')
leap_binder.set_metadata(function=all_raw_metadata, name='all_raw_metadata')
leap_binder.set_visualizer(function=text_visualizer_func, visualizer_type=LeapDataType.Text,
                           name='text_from_token_input')
leap_binder.set_visualizer(function=text_visualizer_output, visualizer_type=LeapDataType.Text, name='gt_text')
leap_binder.add_prediction(name='sentiment', labels=['positive', 'negative'])
