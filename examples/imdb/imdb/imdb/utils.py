"""
Standartization is taken from the example at https://www.tensorflow.org/tutorials/keras/text_classification
"""

from os import listdir, getcwd
from os.path import isfile, join, splitext, basename, dirname
import json
import re
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from keras_preprocessing.text import Tokenizer as TokenizerType
import textstat
from pathlib import Path
from typing import Tuple, Any, List


def create_json_for_imdb(imdb_folder_path: str) -> None:
    POS_NAME = "pos"
    NEG_NAME = "neg"
    SUBFOLDERS = ["train", "test"]
    files_dict = {}
    gt_names = [POS_NAME, NEG_NAME]
    for folder in SUBFOLDERS:
        curr_path = join(imdb_folder_path, folder)
        relative_path = join(basename(imdb_folder_path), folder)
        folders = [join(curr_path, name) for name in gt_names]
        files = [[f for f in listdir(foldr) if isfile(join(foldr, f)) and splitext(f)[-1] == ".txt"] for foldr in folders]
        pos_files_relative = [join(relative_path, POS_NAME, f) for f in files[0]]
        neg_files_relative = [join(relative_path, NEG_NAME, f) for f in files[1]]
        files_dict[folder] = {gt_names[0]: pos_files_relative, gt_names[1]: neg_files_relative}
        for i, gt_type_folders in enumerate([pos_files_relative, neg_files_relative]):
            metrics = [None]*len(gt_type_folders)
            for j, fp in enumerate(gt_type_folders):
                print(j)
                comment = load_imdb_comment(join(Path(__file__).parent.parent.parent, fp))
                metadata_names, metadata = compute_metadata(comment)
                metrics[j] = metadata
            files_dict[folder][gt_names[i] + "_metrics"] = metrics
    train_dict = files_dict[SUBFOLDERS[0]]
    train_dict['metrics_titles'] = metadata_names
    test_dict = files_dict[SUBFOLDERS[1]]
    test_dict['metrics_titles'] = metadata_names
    with open("train_dict.json", 'w') as f:
        json.dump(train_dict, f)
    with open("test_dict.json", 'w') as f:
        json.dump(test_dict, f)


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
