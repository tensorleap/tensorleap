import os
from typing import Union, List
import math
import numpy as np

from code_loader.contract.enums import DatasetInputType, DatasetOutputType, DatasetMetadataType
from code_loader.contract.datasetclasses import SubsetResponse
from code_loader import dataset_binder
from google.cloud import storage
from google.cloud.storage import Bucket
from google.auth.credentials import AnonymousCredentials
from os.path import join
from keras_preprocessing.text import Tokenizer as TokenizerType
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
import json
from os.path import basename, dirname
from typing import List, Optional, Callable
import re
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences
NUMBER_OF_SAMPLES = 20000
crop_size = 128
BUCKET_NAME = 'example-datasets-47ml982d'
PROJECT_ID = 'example-dev-project-nmrksf0o'
METRIC_NAMES = ["flesch_reading", "flesch_kincaid", "coleman_liau_index", "automated_readability_index",
                "dale_chall_readability_score", "difficult_words", "linsear_write_formula",
                "gunning_fog", "fernandez_huerta", "szigriszt_pazos",
                "gutierrez_polini", "crawford", "gulpease_index", "osman"]


def standartize(comment: str) -> str:
    print("standartize")
    lowercase = comment.lower()
    html_stripped = re.sub('<br />', ' ', lowercase)
    punctuation_stripped = re.sub('[%s]' % re.escape(string.punctuation), '', html_stripped)
    return punctuation_stripped


def prepare_input(tokanizer: TokenizerType, input_text: str, sequence_length: int = 250) -> np.ndarray:
    print("prepare_input")
    standard_text = standartize(input_text)
    tokanized_input = tokanizer.texts_to_sequences([standard_text])
    padded_input = pad_sequences(tokanized_input, maxlen=sequence_length)
    return padded_input[0, ...]


def get_gt(samples_path: List[str]) -> List[float]:
    print("get_gt")
    gt_list = [None]*len(samples_path)
    for i in range(len(samples_path)):
        if basename(dirname(samples_path[i])) == "pos":
            gt_list[i] = [1.0, 0.]
        else:
            gt_list[i] = [0., 1.0]
    return gt_list


def load_tokanizer(tokanizer_path: str) -> TokenizerType:
    print("load_tokanizer")
    with open(tokanizer_path, 'r') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    print("finished with loading tokanizer")
    return tokenizer


def download_load_assets() -> (TokenizerType, dict):
    print("download_load_assets")
    cloud_path = join("assets", "train_dict.json")
    local_path = _download(cloud_path)
    with open(local_path, 'r') as f:
        train_dict = json.load(f)
    print(len(train_dict['pos']))
    cloud_path = join("assets", "tokenizer.json")
    local_path = _download(cloud_path)
    tokenizer = load_tokanizer(local_path)
    print("finished downloading assets")
    return tokenizer, train_dict


def create_input(train_dict: dict, half_t_size: int, half_v_size: int) -> (List[str], List[List[float]],
                                                                           List[List[float]], List[str],
                                                                           List[List[float]], List[List[float]]):
    print("create_input")
    train_paths = train_dict['pos'][:half_t_size]+train_dict['neg'][:half_t_size]
    train_gt = get_gt(train_paths)
    train_metrics = train_dict['pos_metrics'][:half_t_size]+train_dict['neg_metrics'][:half_t_size]
    val_paths = train_dict['pos'][half_t_size:half_t_size+half_v_size] + \
                train_dict['neg'][half_t_size:half_t_size+half_v_size]
    val_gt = get_gt(val_paths)
    val_metrics = train_dict['pos_metrics'][half_t_size:half_t_size+half_v_size] + \
                  train_dict['neg_metrics'][half_t_size:half_t_size+half_v_size]
    return train_paths, train_gt, train_metrics, val_paths, val_gt, val_metrics


def _connect_to_gcs() -> Bucket:
    print("_connect_to_gcs")
    gcs_client = storage.Client(project=PROJECT_ID, credentials=AnonymousCredentials())
    return gcs_client.bucket(BUCKET_NAME)


def _download(cloud_file_path: str, local_file_path: Optional[str] = None) -> str:
    BASE_PATH = "imdb"
    cloud_file_path = join(BASE_PATH, cloud_file_path)
    print("download")
    # if local_file_path is not specified saving in home dir
    if local_file_path is None:
        home_dir = os.getenv("HOME")
        local_file_path = os.path.join(home_dir, "Tensorleap_data", BUCKET_NAME, cloud_file_path)
    print(local_file_path)
    # check if file already exists
    if os.path.exists(local_file_path):
        return local_file_path

    bucket = _connect_to_gcs()
    dir_path = os.path.dirname(local_file_path)
    os.makedirs(dir_path, exist_ok=True)
    blob = bucket.blob(cloud_file_path)
    blob.download_to_filename(local_file_path)
    return local_file_path


def subset_func() -> List[SubsetResponse]:
    print("subset_func")
    tokenizer, train_dict = download_load_assets()
    dataset_binder.cache_container["word_to_index"]["tokens"] = tokenizer.word_index
    max_size = 2*len(train_dict['pos'])
    train_size = min(NUMBER_OF_SAMPLES, int(0.9*max_size))
    half_t_size = int(train_size/2)
    val_size = int(0.1*train_size)
    half_v_size = int(val_size/2)
    train_paths, train_gt, train_metrics, val_paths, val_gt, val_metrics\
        = create_input(train_dict, half_t_size, half_v_size)
    train = SubsetResponse(length=2*half_t_size, data={'paths': train_paths, 'gt': train_gt, 'tokenizer': tokenizer,
                                                       'metrics': train_metrics})
    val = SubsetResponse(length=2*half_v_size, data={'paths': val_paths, 'gt': val_gt, 'tokenizer': tokenizer,
                                                     'metrics': val_metrics})
    response = [train, val]
    return response


def input_tokens(idx: int, subset: SubsetResponse) -> np.ndarray:
    print("input_tokens")
    comment_path = subset.data['paths'][idx]
    local_path = _download(comment_path)
    with open(local_path, 'r') as f:
        comment = f.read()
    tokenizer = subset.data['tokenizer']
    padded_input = prepare_input(tokenizer, comment)
    return padded_input


def gt_sentiment(idx: int, subset: Union[SubsetResponse, list]) -> List[float]:
    print("gt_sentiment")
    return subset.data['gt'][idx]


def metadata_path(idx: int, subset: Union[SubsetResponse, list]) -> str:
    print("metadata_path")
    return subset.data['paths'][idx]


def metadata_encoder(metric_idx: int) -> Callable[[int, Union[SubsetResponse, list]], float]:
    def func(idx: int, subset: Union[SubsetResponse, list]) -> float:
        return subset.data['metrics'][idx][metric_idx]
    func.__name__ = METRIC_NAMES[metric_idx]
    return func


dataset_binder.set_subset(function=subset_func, name='IMDBComments')

dataset_binder.set_input(function=input_tokens, subset='IMDBComments', input_type=DatasetInputType.Time_series,
                         name='tokens')

dataset_binder.set_ground_truth(function=gt_sentiment, subset='IMDBComments',
                                ground_truth_type=DatasetOutputType.Classes,
                                name='sentiment', labels=['positive', 'negative'], masked_input=None)

dataset_binder.set_metadata(function=metadata_path, subset='IMDBComments',
                                metadata_type=DatasetMetadataType.string,
                                name='label')

for i in range(len(METRIC_NAMES)):
    dataset_binder.set_metadata(function=metadata_encoder(i), subset='IMDBComments',
                                metadata_type=DatasetMetadataType.float,
                                name=METRIC_NAMES[i])

