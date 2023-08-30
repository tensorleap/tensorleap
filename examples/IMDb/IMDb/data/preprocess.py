
import json
from keras.preprocessing.text import tokenizer_from_json
import pandas as pd
from os.path import join

from IMDb.config import CONFIG
from IMDb.gcs_utils import _download
from transformers import AutoTokenizer

def load_tokenizer(tokenizer_path: str):
    with open(tokenizer_path, 'r') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    return tokenizer


def download_load_assets():
    cloud_path = join("assets", "imdb.csv")
    local_path = _download(cloud_path)
    df = pd.read_csv(local_path)
    cloud_path = join("assets", "tokenizer_v2.json")
    local_path = _download(cloud_path)
    # tokenizer = load_tokenizer(local_path)
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['MODEL_NAME'])
    return tokenizer, df