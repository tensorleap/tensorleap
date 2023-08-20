from typing import List, Optional, Callable, Tuple, Dict

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
# the PreprocessResponse's data. Returns a numpy array containing padded tokenized input.
def input_tokens(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    comment_path = preprocess.data['df']['paths'][idx]
    local_path = _download(comment_path)
    with open(local_path, 'r') as f:
        comment = f.read()
    tokenizer = preprocess.data['tokenizer']
    padded_input = prepare_input(tokenizer, comment)
    return padded_input


# Ground Truth Encoder - fetches the label with the index `idx` from the `gt` array set in
# the PreprocessResponse's  data. Returns a numpy array containing a hot vector label correlated with the sample.
def gt_sentiment(idx: int, preprocess: PreprocessResponse) -> List[float]:
    gt_str = preprocess.data['df']['gt'][idx]
    return preprocess.data['ohe'][gt_str]


# Metadata functions allow to add extra data for a later use in analysis.
# This metadata adds the ground truth of each sample (not a hot vector).
def gt_metadata(idx: int, preprocess: PreprocessResponse) -> str:
    if preprocess.data['df']['gt'][idx] == "pos":
        return "positive"
    else:
        return "negative"


# Visualizer functions define how to interpet the data and visualize it.
# In this example we define a tokens-to-text visualizer.
def text_visualizer_func(data: np.ndarray) -> LeapText:
    tokenizer = leap_binder.custom_tokenizer
    texts = tokenizer.sequences_to_texts([data])
    return LeapText(texts[0].split(' '))


# Binders
leap_binder.set_preprocess(function=preprocess_func)
leap_binder.set_input(function=input_tokens, name='tokens')
leap_binder.set_ground_truth(function=gt_sentiment, name='sentiment')
leap_binder.set_metadata(function=gt_metadata, metadata_type=DatasetMetadataType.string, name='gt')
leap_binder.set_visualizer(function=text_visualizer_func, visualizer_type=LeapDataType.Text, name='text_from_token')
leap_binder.add_prediction(name='sentiment', labels=['positive', 'negative'])
