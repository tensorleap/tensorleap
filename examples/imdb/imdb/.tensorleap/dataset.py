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
from typing import List, Optional, Callable, Tuple, Dict, Any
import re
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences
NUMBER_OF_SAMPLES = 20000
BUCKET_NAME = 'example-datasets-47ml982d'
PROJECT_ID = 'example-dev-project-nmrksf0o'
METRIC_NAMES = ["flesch_reading", "flesch_kincaid", "coleman_liau_index", "automated_readability_index",
                "dale_chall_readability_score", "difficult_words", "linsear_write_formula",
                "gunning_fog", "fernandez_huerta", "szigriszt_pazos",
                "gutierrez_polini", "crawford", "gulpease_index", "osman"]


def standartize(comment: str) -> str:
    lowercase = comment.lower()
    html_stripped = re.sub('<br />', ' ', lowercase)
    punctuation_stripped = re.sub('[%s]' % re.escape(string.punctuation), '', html_stripped)
    return punctuation_stripped


def prepare_input(tokanizer: TokenizerType, input_text: str, sequence_length: int = 250) -> np.ndarray:
    standard_text = standartize(input_text)
    tokanized_input = tokanizer.texts_to_sequences([standard_text])
    padded_input = pad_sequences(tokanized_input, maxlen=sequence_length)
    return padded_input[0, ...]


def get_gt(samples_path: List[str]) -> List[List[float]]:
    # gt_list = [None]*len(samples_path)
    gt_list = np.zeros((len(samples_path), 2), dtype=float).tolist()
    for i in range(len(samples_path)):
        if basename(dirname(samples_path[i])) == "pos":
            gt_list[i] = [1.0, 0.]
        else:
            gt_list[i] = [0., 1.0]
    return gt_list


def load_tokanizer(tokanizer_path: str) -> TokenizerType:
    with open(tokanizer_path, 'r') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    return tokenizer


def download_load_assets() -> Tuple[TokenizerType, Dict[str, np.ndarray]]:
    cloud_path = join("assets", "train_dict_v5.json")
    local_path = _download(cloud_path)
    with open(local_path, 'r') as f:
        train_dict = json.load(f)
    cloud_path = join("assets", "tokenizer_v2.json")
    local_path = _download(cloud_path)
    tokenizer = load_tokanizer(local_path)
    return tokenizer, train_dict


def _connect_to_gcs() -> Bucket:
    gcs_client = storage.Client(project=PROJECT_ID, credentials=AnonymousCredentials())
    return gcs_client.bucket(BUCKET_NAME)


def _download(cloud_file_path: str, local_file_path: Optional[str] = None) -> str:
    BASE_PATH = "imdb"
    cloud_file_path = join(BASE_PATH, cloud_file_path)
    # if local_file_path is not specified saving in home dir
    if local_file_path is None:
        home_dir = os.getenv("HOME")
        assert home_dir is not None
        local_file_path = os.path.join(home_dir, "Tensorleap_data", BUCKET_NAME, cloud_file_path)
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
    tokenizer, train_dict = download_load_assets()
    dataset_binder.cache_container["word_to_index"]["tokens"] = tokenizer.word_index
    combined_size = 2*len(train_dict['pos'])
    truncated_size = min(NUMBER_OF_SAMPLES, int(0.9*combined_size))
    train_label_size = int(truncated_size/2)
    truncated_val_size = int(0.1*truncated_size)
    val_label_size = int(truncated_val_size/2)
    input_keys_pre = ['', 'metrics', 'length', 'oov', 'polarity', 'subjectivity']
    output_keys = ['paths', 'metrics', 'length', 'oov_count', 'polarity', 'subjectivity']
    gt = ['pos', 'neg']
    data_dict = {'train': {}, 'val': {}}
    # Building the dictionary for the Subset response - composed of "[POS/NEG]_METRICNAME"
    for in_key_pre, out_key in zip(input_keys_pre, output_keys):
        if in_key_pre != '':
            # Accessing the [POS/NEG]_METRICNAME key in the preprocessed data
            in_key = [gt[j] + "_" + in_key_pre for j in range(len(gt))]
        else:
            in_key = gt
        data_dict['train'][out_key] = train_dict[in_key[0]][:train_label_size] + \
                                            train_dict[in_key[1]][:train_label_size]
        data_dict['val'][out_key] = train_dict[in_key[0]][train_label_size:train_label_size+val_label_size] +\
                                      train_dict[in_key[1]][train_label_size:train_label_size+val_label_size]
    data_dict['train']['gt'] = get_gt(data_dict['train']['paths'])
    data_dict['val']['gt'] = get_gt(data_dict['val']['paths'])
    data_dict['train']['tokenizer'] = tokenizer
    data_dict['val']['tokenizer'] = tokenizer
    # Our train/val are composed of 50% negative and 50% positive samples
    train = SubsetResponse(length=2*train_label_size, data=data_dict['train'])
    val = SubsetResponse(length=2*val_label_size, data=data_dict['val'])
    response = [train, val]
    return response


# Input Encoder
def input_tokens(idx: int, subset: SubsetResponse) -> np.ndarray:
    comment_path = subset.data['paths'][idx]
    local_path = _download(comment_path)
    with open(local_path, 'r') as f:
        comment = f.read()
    tokenizer = subset.data['tokenizer']
    padded_input = prepare_input(tokenizer, comment)
    return padded_input


# GT Encoder
def gt_sentiment(idx: int, subset: SubsetResponse) -> List[float]:
    return subset.data['gt'][idx]


# Metadata Encoders
def gt_metadata(idx: int, subset: SubsetResponse) -> str:
    if subset.data['gt'][idx][0] == 1.0:
        return "positive"
    else:
        return "negative"


def metadata_encoder(metric_idx: int) -> Callable[[int, SubsetResponse], float]:
    def func(idx: int, subset: SubsetResponse) -> float:
        return subset.data['metrics'][idx][metric_idx]
    func.__name__ = METRIC_NAMES[metric_idx]
    return func


def length_metadata(idx: int, subset: SubsetResponse) -> int:
    return subset.data['length'][idx]


def oov_metadata(idx: int, subset: SubsetResponse) -> int:
    return subset.data['oov_count'][idx]


def score_metadata(idx, subset: SubsetResponse) -> int:
    return int(subset.data['paths'][idx].split("_")[1].split(".")[0])


def score_confidence_metadata(idx, subset: SubsetResponse) -> int:
    return abs(5 - int(subset.data['paths'][idx].split("_")[1].split(".")[0]))


def polarity_metadata(idx, subset: SubsetResponse) -> float:
    return subset.data['polarity'][idx]


def subjectivity_metadata(idx, subset: SubsetResponse) -> float:
    return subset.data['subjectivity'][idx]


# Binders
dataset_binder.set_subset(function=subset_func, name='IMDBComments')


dataset_binder.set_input(function=input_tokens, subset='IMDBComments', input_type=DatasetInputType.Text,
                         name='tokens')

dataset_binder.set_ground_truth(function=gt_sentiment, subset='IMDBComments',
                                ground_truth_type=DatasetOutputType.Classes,
                                name='sentiment', labels=['positive', 'negative'], masked_input=None)

dataset_binder.set_metadata(function=gt_metadata, subset='IMDBComments',
                                metadata_type=DatasetMetadataType.string,
                                name='gt')

dataset_binder.set_metadata(function=oov_metadata, subset='IMDBComments',
                                metadata_type=DatasetMetadataType.int,
                                name='oov_count')

dataset_binder.set_metadata(function=length_metadata, subset='IMDBComments',
                                metadata_type=DatasetMetadataType.int,
                                name='length')

dataset_binder.set_metadata(function=score_metadata, subset='IMDBComments',
                                metadata_type=DatasetMetadataType.int,
                                name='score')

dataset_binder.set_metadata(function=score_confidence_metadata, subset='IMDBComments',
                                metadata_type=DatasetMetadataType.int,
                                name='score_confidence')

dataset_binder.set_metadata(function=polarity_metadata, subset='IMDBComments',
                                metadata_type=DatasetMetadataType.float,
                                name='polarity')

dataset_binder.set_metadata(function=subjectivity_metadata, subset='IMDBComments',
                                metadata_type=DatasetMetadataType.float,
                                name='subjectivity')

for i in range(len(METRIC_NAMES)):
    dataset_binder.set_metadata(function=metadata_encoder(i), subset='IMDBComments',
                                metadata_type=DatasetMetadataType.float,
                                name=METRIC_NAMES[i])

