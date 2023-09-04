from typing import List, Dict
import pandas as pd
import numpy as np
import numpy.typing as npt
import tensorflow as tf

# Tensorleap imports
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.enums import LeapDataType
from code_loader.contract.visualizer_classes import LeapText, LeapHorizontalBar

from IMDb.config import CONFIG
from IMDb.data.preprocess import download_load_assets
from IMDb.gcs_utils import _download
from IMDb.utils import prepare_input, prepare_input_dense_model


# Preprocess Function
def preprocess_func() -> List[PreprocessResponse]:
    """
    Performs data preprocessing and prepares training and validation data for a machine learning model.

    :return: A list of PreprocessResponse objects containing preprocessed data and associated information.
    """
    tokenizer, df = download_load_assets()
    train_label_size = int(0.9 * CONFIG['NUMBER_OF_SAMPLES'] / 2)
    val_label_size = int(0.1 * CONFIG['NUMBER_OF_SAMPLES'] / 2)
    df = df[df['subset'] == 'train']
    train_df = pd.concat([df[df['gt'] == 'pos'][:train_label_size], df[df['gt'] == 'neg'][:train_label_size]],
                         ignore_index=True)
    val_df = pd.concat([df[df['gt'] == 'pos'][train_label_size:train_label_size + val_label_size],
                        df[df['gt'] == 'neg'][train_label_size:train_label_size + val_label_size]], ignore_index=True)
    ohe = {"pos": [0., 1.0], "neg": [1.0, 0.]}

    train = PreprocessResponse(length=2 * train_label_size, data={"df": train_df, "tokenizer": tokenizer, "ohe": ohe})
    val = PreprocessResponse(length=2 * val_label_size, data={"df": val_df, "tokenizer": tokenizer, "ohe": ohe})
    response = [train, val]
    leap_binder.custom_tokenizer = tokenizer

    return response


def input_func(idx: int, preprocess: PreprocessResponse):
    """
    Reads and preprocesses an input comment for machine learning model input.

    :param idx: The index of the comment to be processed.
    :param preprocess: preprocessed data.
    :return: A padded and tokenized input for the model.
    """
    comment_path = preprocess.data['df']['paths'][idx]
    local_path = _download(comment_path)
    with open(local_path, 'r') as f:
        comment = f.read()
    tokenizer = preprocess.data['tokenizer']
    padded_input = prepare_input(tokenizer, comment)
    return padded_input


def input_tokens(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    """
    Retrieves and returns the input tokens for a specific comment after preprocessing.

    :param idx: The index of the comment for which tokens are retrieved.
    :param preprocess: preprocessed data.
    :return: A NumPy array containing the preprocessed input tokens.
    """
    padded_input = input_func(idx, preprocess)
    padded_input = padded_input.squeeze()
    return padded_input


def input_ids(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    """
    Retrieves and returns the input IDs for a specific comment after preprocessing.

    :param idx: The index of the comment for which input IDs are retrieved.
    :param preprocess: preprocessed data.
    :return: A NumPy array containing the preprocessed input IDs.
    """
    padded_input = input_func(idx, preprocess)
    padded_input = np.array(padded_input['input_ids'])
    padded_input = padded_input.squeeze()
    return padded_input


def attention_masks(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    """
    Retrieves and returns the attention masks for a specific comment after preprocessing.

    :param idx: The index of the comment for which attention masks are retrieved.
    :param preprocess: preprocessed data.
    :return: A NumPy array containing the preprocessed attention masks.
    """
    padded_input = input_func(idx, preprocess)
    padded_input = np.array(padded_input['attention_mask'])
    padded_input = padded_input.squeeze()
    return np.array(padded_input)


def token_type_ids(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    """
    Extracts and returns the token type IDs for a specific comment after preprocessing.

    :param idx: The index of the comment for which token type IDs are extracted.
    :param preprocess: preprocessed data.
    :return: A NumPy array containing the extracted token type IDs.
    """
    padded_input = input_func(idx, preprocess)
    padded_input = np.array(padded_input['token_type_ids'])
    padded_input = padded_input.squeeze()
    return np.array(padded_input)


def gt_sentiment(idx: int, preprocess: PreprocessResponse) -> List[float]:
    """
    :param idx: The index of the comment for which sentiment values are retrieved.
    :param preprocess: preprocessed data.
    :return: A list of float values representing the sentiment of the comment.
    """
    gt_str = preprocess.data['df']['gt'][idx]
    return preprocess.data['ohe'][gt_str]


def gt_metadata(idx: int, preprocess: PreprocessResponse) -> str:
    """
    Retrieves and returns the sentiment metadata (positive or negative) for a specific comment.

    :param idx: The index of the comment for which metadata is retrieved.
    :param preprocess: preprocessed data.
    :return: A string representing the sentiment metadata.
    """
    if preprocess.data['df']['gt'][idx] == "pos":
        return "positive"
    else:
        return "negative"


def all_raw_metadata(idx: int, preprocess: PreprocessResponse) -> Dict:
    """
    Retrieves and returns various raw metadata values for a specific comment.

    :param idx: The index of the comment for which raw metadata is retrieved.
    :param preprocess: A PreprocessResponse object containing preprocessing information.
    :return: A dictionary containing raw metadata values.
    """
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
    Decodes the input tokens from their corresponding input IDs using the provided tokenizer.
    :param tokenizer: The tokenizer used to convert token IDs to text.
    :param input_ids: (np.ndarray): Array of input token IDs.
    :returns: decoded (List[str]): List of decoded tokens as strings.
    """
    decoded = tokenizer.decode(input_ids)
    return decoded


def pad_list(input_list: List, desired_length: int = CONFIG['SEQUENCE_LENGTH'], padding_element: str = "[PAD]"):
    """
    Pads a given list to the desired length with a specified padding element.

    :param input_list: The input list to be padded.
    :param desired_length: The desired length of the padded list.
    :param padding_element: The element used for padding.
    :return: The padded list with the specified length.
    """
    current_length = len(input_list)

    if current_length >= desired_length:
        return input_list

    padding_needed = desired_length - current_length
    padded_list = input_list + [padding_element] * padding_needed

    return padded_list


def text_visualizer_func(input_ids: np.ndarray) -> LeapText:
    """
    Converts input token IDs into a LeapText object for visualization.

    :param input_ids: A NumPy array containing input token IDs.
    :return: A LeapText object representing the tokenized and padded text for visualization.
    """
    tokenizer = leap_binder.custom_tokenizer
    data = input_ids.astype(np.int64)
    text = tokenizer_decoder(tokenizer, data)
    text_input = [token for token in text.split()]
    padded_list = pad_list(text_input)
    return LeapText(padded_list)


def text_visualizer_func_dense_model(data: np.ndarray) -> LeapText:
    """
    Converts input data from a dense model into a LeapText object for visualization.

    :param data: A NumPy array containing input data from a dense model.
    :return: A LeapText object representing the tokenized and padded text for visualization.
    """
    tokenizer = leap_binder.custom_tokenizer
    texts = tokenizer.sequences_to_texts([data])
    text_input = texts[0].split(' ')
    text_input = [text for text in text_input]
    padded_list = pad_list(text_input)
    return LeapText(padded_list)

def horizontal_bar_visualizer_with_labels_name(y_pred: npt.NDArray[np.float32]) -> LeapHorizontalBar:
    labels_names = [CONFIG['LABELS_NAMES'][index] for index in range(y_pred.shape[-1])]
    return LeapHorizontalBar(y_pred, labels_names)


# Binders
leap_binder.set_preprocess(function=preprocess_func)
leap_binder.set_input(function=input_ids, name='input_ids')
leap_binder.set_input(function=attention_masks, name='attention_masks')
leap_binder.set_input(function=token_type_ids, name='token_type_ids')
leap_binder.set_ground_truth(function=gt_sentiment, name='sentiment')
leap_binder.set_metadata(function=gt_metadata, name='gt')
leap_binder.set_metadata(function=all_raw_metadata, name='all_raw_metadata')
leap_binder.set_visualizer(function=text_visualizer_func, visualizer_type=LeapDataType.Text,
                           name='text_from_token_input')
leap_binder.set_visualizer(function=horizontal_bar_visualizer_with_labels_name,
                           visualizer_type=LeapDataType.HorizontalBar, name='pred_labels')
leap_binder.add_prediction(name='sentiment', labels=['positive', 'negative'])

if __name__ == '__main__':
    leap_binder.check()
