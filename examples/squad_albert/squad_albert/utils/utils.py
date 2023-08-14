from typing import List
import numpy as np
import nltk
import tensorflow as tf

from squad_albert.config import CONFIG

nltk.download('punkt')
np.random.seed(0)
#tokenizer = AlbertTokenizerFast.from_pretrained("vumichien/albert-base-v2-squad2")


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
    context_end = CONFIG['max_sequence_length'] - tf.argmax(token_type_ids[::-1]) - 1
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









