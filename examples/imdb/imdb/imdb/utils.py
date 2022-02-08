"""
Standartization is taken from the example at https://www.tensorflow.org/tutorials/keras/text_classification
Attention module is adapted from https://keras.io/examples/nlp/text_classification_with_transformer/
positional embedding is adapted from https://keras.io/examples/nlp/masked_language_modeling/
"""

from os import listdir, getcwd
from os.path import isfile, join, splitext, basename, dirname
import json
import re
import string

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras_preprocessing.text import Tokenizer as TokenizerType
import textstat
from pathlib import Path
from tensorflow.keras import layers


def create_json_for_imdb(imdb_folder_path: str, tokenizer_path: str) -> None:
    POS_NAME = "pos"
    NEG_NAME = "neg"
    SUBFOLDERS = ["train", "test"]
    files_dict = {}
    gt_names = [POS_NAME, NEG_NAME]
    tokenizer = load_tokanizer(tokenizer_path)
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
            lengths = np.zeros(len(gt_type_folders), dtype=int)
            for j, fp in enumerate(gt_type_folders):
                print(j)
                comment = load_imdb_comment(join(Path(__file__).parent.parent.parent, fp))
                standard_comment = standartize(comment)
                tokenized_comment = tokenizer.texts_to_sequences([standard_comment])
                comment_length = len(tokenized_comment[0])
                lengths[j] = comment_length
                metadata_names, metadata = compute_metadata(comment)
                metrics[j] = metadata
            files_dict[folder][gt_names[i] + "_metrics"] = metrics
            files_dict[folder]['length'] = lengths
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


def load_tokanizer(tokanizer_path: str) -> TokenizerType:
    print("load_tokanizer")
    with open(tokanizer_path, 'r') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    print("finished with loading tokanizer")
    return tokenizer


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


def compute_metadata(comment: str) -> (list, tuple):
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


class TransformerBlock:
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn1 = layers.Dense(ff_dim, activation='relu')
        self.ffn2 = layers.Dense(embed_dim)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.add_1 = layers.Add()
        self.add_2 = layers.Add()
        self.add_3 = layers.Add()

    def call(self, inputs, attention_mask):
        attn_output = self.att(inputs, inputs, attention_mask=attention_mask)
        attn_output = self.dropout1(attn_output)
        added = self.add_1([inputs, attn_output])
        out1 = self.layernorm1(added)
        ffn_output = self.ffn1(out1)
        ffn_output = self.ffn2(ffn_output)
        ffn_output = self.dropout2(ffn_output)
        transformer_out = self.add_3([out1, ffn_output])
        return transformer_out


class TokenAndPositionEmbedding:
    def __init__(self, maxlen, vocab_size, embed_dim):
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.add_layer = layers.Add()
        self.max_len = maxlen

    def call(self, tokens, positions):
        positions = self.pos_emb(positions)
        x = self.token_emb(tokens)
        embeddings_out = self.add_layer([x, positions])
        return embeddings_out


def temp_lengths(imdb_folder_path: str, tokenizer_path: str, train_dict_p: dict, test_dict_p: dict) -> None:
    POS_NAME = "pos"
    NEG_NAME = "neg"
    SUBFOLDERS = ["train", "test"]
    gt_names = [POS_NAME, NEG_NAME]
    with open(train_dict_p, 'r') as f:
        train_dict = json.load(f)
    with open(test_dict_p, 'r') as f:
        test_dict = json.load(f)
    dicts = [train_dict, test_dict]
    tokenizer = load_tokanizer(tokenizer_path)
    for k, folder in enumerate(SUBFOLDERS):
        curr_path = join(imdb_folder_path, folder)
        relative_path = join(basename(imdb_folder_path), folder)
        folders = [join(curr_path, name) for name in gt_names]
        files = [[f for f in listdir(foldr) if isfile(join(foldr, f)) and splitext(f)[-1] == ".txt"] for foldr in folders]
        pos_files_relative = [join(relative_path, POS_NAME, f) for f in files[0]]
        neg_files_relative = [join(relative_path, NEG_NAME, f) for f in files[1]]
        for i, gt_type_folders in enumerate([pos_files_relative, neg_files_relative]):
            lengths = [None]*len(gt_type_folders)
            for j, fp in enumerate(gt_type_folders):
                print(j)
                comment = load_imdb_comment(join(Path(__file__).parent.parent.parent, fp))
                standard_comment = standartize(comment)
                tokenized_comment = tokenizer.texts_to_sequences([standard_comment])
                comment_length = len(tokenized_comment[0])
                lengths[j] = comment_length
            dicts[k][gt_names[i] + "_length"] = lengths
    with open("train_dict_v2.json", 'w') as f:
        json.dump(train_dict, f)
    with open("test_dict_v2.json", 'w') as f:
        json.dump(test_dict, f)


def get_pos_encoding_matrix(max_len, d_emb):
    pos_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
            if pos != 0
            else np.zeros(d_emb)
            for pos in range(max_len)
        ]
    )
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc
#
#
# temp_lengths("/home/tomtensor/Work/Projects/examples/tensorleap/examples/imdb/aclImdb",
#              "tokenizer.json",
#              "train_dict.json",
#              "test_dict.json")