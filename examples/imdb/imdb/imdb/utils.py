"""
Standartization is taken from the example at https://www.tensorflow.org/tutorials/keras/text_classification
"""

from os import listdir, getcwd
from os.path import isfile, join, splitext, basename, dirname
import json
import re
import string
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras_preprocessing.text import Tokenizer as TokenizerType
import textstat
from pathlib import Path
from typing import Tuple, Any, List
from textblob import TextBlob
from pandas import DataFrame
import numpy as np


def create_csv_for_imdb(imdb_folder_path: str) -> None:
    POS_NAME = "pos"
    NEG_NAME = "neg"
    SUBFOLDERS = ["train", "test"]
    gt_names = [POS_NAME, NEG_NAME]
    tokenizer = load_tokanizer("tokenizer_v2.json")
    df = DataFrame()
    for folder in SUBFOLDERS:
        curr_path = join(imdb_folder_path, folder)
        relative_path = join(basename(imdb_folder_path), folder)
        folders = [join(curr_path, name) for name in gt_names]
        files = [[f for f in listdir(foldr) if isfile(join(foldr, f)) and splitext(f)[-1] == ".txt"] for foldr in folders]
        pos_files_relative = [join(relative_path, POS_NAME, f) for f in files[0]]
        neg_files_relative = [join(relative_path, NEG_NAME, f) for f in files[1]]
        paths = [pos_files_relative, neg_files_relative]
        for i, gt_type_folders in enumerate(paths):
            metrics = [None]*len(gt_type_folders)
            lengths = [None]*len(gt_type_folders)
            oovs = [None]*len(gt_type_folders)
            polarities = [None]*len(gt_type_folders)
            all_subject = [None]*len(gt_type_folders)
            for j, fp in enumerate(gt_type_folders):
                print(j)
                comment = load_imdb_comment(join(Path(__file__).parent.parent.parent, fp))
                standard_comment = standartize(comment)
                blob = TextBlob(standard_comment)
                polarity = blob.polarity
                subjectivity = blob.subjectivity
                tokenized_comment = tokenizer.texts_to_sequences([standard_comment])[0]
                comment_length = len(tokenized_comment)
                lengths[j] = comment_length
                oov = get_oov_count(tokenized_comment)
                metadata_names, metadata = compute_metadata(comment)
                metrics[j] = metadata
                oovs[j] = oov
                polarities[j] = polarity
                all_subject[j] = subjectivity
            curr_dict = {"oov_count": oovs, "length": lengths, "subjectivity": all_subject, "polarity": polarities,
                         "paths": paths[i]}
            curr_dict.update(dict(zip(metadata_names, np.array(metrics).transpose())))
            cur_dataframe = pd.DataFrame(curr_dict)
            cur_dataframe['gt'] = gt_names[i]
            cur_dataframe['subset'] = folder
            df = pd.concat([df, cur_dataframe], ignore_index=True)
    df.to_csv("imdb.csv")


def load_imdb_comment(file: str) -> str:
    with open(file, 'r') as f:
        comment = f.read()
    return comment


def standartize(comment: str) -> str:
    lowercase = comment.lower()
    html_stripped = re.sub('<br />', ' ', lowercase)
    punctuation_stripped = re.sub('[%s]' % re.escape(string.punctuation), '', html_stripped)
    return punctuation_stripped


def fit_save_tokanizer(comment_paths: str) -> TokenizerType:
    max_features = 10000
    str_list = [None]*len(comment_paths)
    for i in range(len(comment_paths)):
        with open(join(dirname(dirname(getcwd())), comment_paths[i]), 'r') as f:
            str_list[i] = standartize(f.read())
    tokenizer = Tokenizer(oov_token="[OOV]", num_words=max_features)
    tokenizer.fit_on_texts(str_list)
    with open("tokenizer.json", 'w') as f:
        f.write(json.dumps(tokenizer.to_json(), ensure_ascii=False))
    return tokenizer


def compute_metadata(comment: str) -> Tuple[List[str], Tuple[Any, ...]]:
    """
    Here we iterate throught the samples and create a metadata dict
    containing multiple text metrics that are used to estimate complexity.
    """
    flesch_reading = textstat.flesch_reading_ease(comment)
    flesch_kincaid = textstat.flesch_kincaid_grade(comment)
    coleman_liau_index = textstat.coleman_liau_index(comment)
    automated_readability_index = textstat.automated_readability_index(comment)
    dale_chall_readability_score = textstat.dale_chall_readability_score(comment)
    difficult_words = textstat.difficult_words(comment)
    linsear_write_formula = textstat.linsear_write_formula(comment)
    gunning_fog = textstat.gunning_fog(comment)
    fernandez_huerta = textstat.fernandez_huerta(comment)
    szigriszt_pazos = textstat.szigriszt_pazos(comment)
    gutierrez_polini = textstat.gutierrez_polini(comment)
    crawford = textstat.crawford(comment)
    gulpease_index = textstat.gulpease_index(comment)
    osman = textstat.osman(comment)
    metric_names = ["flesch_reading", "flesch_kincaid", "coleman_liau_index", "automated_readability_index",
                    "dale_chall_readability_score", "difficult_words", "linsear_write_formula",
                    "gunning_fog", "fernandez_huerta", "szigriszt_pazos",
                    "gutierrez_polini", "crawford", "gulpease_index", "osman"]
    metrics = tuple([flesch_reading, flesch_kincaid, coleman_liau_index, automated_readability_index,
               dale_chall_readability_score, difficult_words, linsear_write_formula, gunning_fog,
               fernandez_huerta, szigriszt_pazos, gutierrez_polini, crawford, gulpease_index, osman])
    return metric_names, metrics


def get_oov_count(tokenized_comment: List[int]) -> int:
    OOV_TOKEN = 1
    oov_count = tokenized_comment.count(OOV_TOKEN)
    return oov_count


def load_tokanizer(tokanizer_path: str) -> TokenizerType:
    with open(tokanizer_path, 'r') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    return tokenizer
