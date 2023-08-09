import numpy as np
import numpy.typing as npt
from typing import List, Tuple
import tensorflow as tf
from textblob import TextBlob

from utils.utils import get_context_positions
from utils.metrices import get_start_end_arrays
from config import CONFIG

def tokens_context_decoder(input_ids: np.ndarray, token_type_ids: np.ndarray, tokenizer)->List[str]:
    """
    Description: Decodes the input context tokens from their corresponding input IDs using the provided tokenizer.
    Parameters:
    input_ids (np.ndarray): Array of input token IDs.
    token_type_ids (np.ndarray): Array of token type IDs indicating the context position.
    tokenizer: The tokenizer used to convert token IDs to text.
    Returns:
    decoded (List[str]): List of context tokens as strings.
    """
    input_ids = input_ids.astype(np.int32).tolist()
    context_start, context_end = get_context_positions(token_type_ids)
    input_ids = input_ids[int(context_start): int(context_end + 1)]
    decoded = tokenizer.decode(input_ids, max_seq_length=CONFIG['max_sequence_length'])
    decoded = decoded.split(' ')
    return decoded

def tokens_question_decoder(input_ids: np.ndarray, token_type_ids: np.ndarray, tokenizer)->List[str]:
    """
    Description: Decodes the input question tokens from their corresponding input IDs using the provided tokenizer.
    Parameters:
    input_ids (np.ndarray): Array of input token IDs.
    token_type_ids (np.ndarray): Array of token type IDs indicating the context position.
    tokenizer: The tokenizer used to convert token IDs to text.
    Returns:
    decoded (List[str]): List of question tokens as strings.
    """
    input_ids = input_ids.astype(np.int32).tolist()
    context_start, context_end = get_context_positions(token_type_ids)
    input_ids = input_ids[1:int(context_start - 1)]
    decoded = tokenizer.decode(input_ids, max_seq_length=CONFIG['max_sequence_length'])
    decoded = decoded.split(' ')
    return decoded

def tokens_decoder(decoded) ->List[str]:
    """
    Description: Truncates or pads the decoded tokens to match the maximum sequence length.
    Parameters:
    decoded (List[str]): List of decoded tokens.
    Returns:
    decoded (List[str]): List of truncated or padded tokens.
    """
    if len(decoded) < CONFIG['max_sequence_length']:  # pad
        decoded += (CONFIG['max_sequence_length'] - len(decoded)) * [CONFIG['PAD_TOKEN']]
    elif len(decoded) > CONFIG['max_sequence_length']:  # truncate
        decoded = decoded[:CONFIG['max_sequence_length']]
    return decoded

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
    decoded = decoded.split(' ')
    return decoded

def answer_decoder(logits: np.ndarray, input_ids: np.ndarray, tokenizer) ->str:
    """
    Description: Decodes the predicted answer from the logits using the provided tokenizer.
    Parameters:
    logits (np.ndarray): Array of logits for start and end indices.
    input_ids (np.ndarray): Array of input token IDs.
    tokenizer: The tokenizer used to convert token IDs to text.
    Returns:
    answer (str): The decoded predicted answer.
    """
    start_logits, end_logits = get_start_end_arrays(logits)
    input_ids = input_ids.astype(np.int32).tolist()
    start_index = int(tf.math.argmax(start_logits, axis=-1))
    end_index = int(tf.math.argmax(end_logits, axis=-1))
    selected_answer_ids = input_ids[start_index:end_index + 1]
    answer = tokenizer_decoder(tokenizer, selected_answer_ids)
    return answer

def context_subjectivity(text: str) -> int:
    """
    Description: Calculates the subjectivity score of the input text using TextBlob.
    Parameters:
    text (str): Input text for subjectivity analysis.
    Returns:
    val (int): Subjectivity score of the text. If the score is not available, returns -1.
    """
    blob = TextBlob(text)
    val = blob.subjectivity
    if val is None:
        val = -1
    return val

def context_polarity(text: str) -> int:
    """
    Description: Calculates the polarity score of the input text using TextBlob.
    Parameters:
    text (str): Input text for polarity analysis.
    Returns:
    val (int): Polarity score of the text. If the score is not available, returns -1.
    """
    blob = TextBlob(text)
    val = blob.polarity
    if val is None:
        val = -1
    return val

def get_decoded_tokens(input_ids: np.ndarray, tokenizer) -> List[str]:
    """
    Description: Decodes the input tokens from their corresponding input IDs using the provided tokenizer and handles special tokens.
    Parameters:
    input_ids (np.ndarray): Array of input token IDs.
    tokenizer: The tokenizer used to convert token IDs to text.
    Returns:
    decoded (List[str]): List of decoded tokens as strings.
    """
    input_ids = input_ids.astype(np.int32).tolist()
    decoded = tokenizer.convert_ids_to_tokens(input_ids)
    ind = decoded.index('<pad>') if '<pad>' in decoded else None
    decoded = decoded[:ind] if ind is not None else decoded  # truncate
    decoded = [token.replace(chr(9601), '') for token in decoded]
    return decoded

def segmented_tokens_decoder(input_ids: np.ndarray, token_type_ids: np.ndarray, gt_logits: np.ndarray, pred_logits: np.ndarray) -> Tuple[npt.NDArray[np.uint8], List[str], List[str]]:
    """
    Description: Decodes the segmented tokens and returns the mask, text, and corresponding labels for visualization and analysis purposes.
    Parameters:
    input_ids (np.ndarray): Array of input token IDs.
    token_type_ids (np.ndarray): Array of token type IDs indicating the context position.
    gt_logits (np.ndarray): Array of logits for the ground truth start and end indices.
    pred_logits (np.ndarray): Array of logits for the predicted start and end indices.
    Returns:
    mask (npt.NDArray[np.uint8]): Mask representing the segmentation labels for each token.
    text (List[str]): List of decoded tokens as strings.
    labels (List[str]): List of segmentation labels.
    """
    mask: npt.NDArray[np.uint8] = np.zeros(len(input_ids))
    labels_mapping = {'other': 0,
                      'question': 1,
                      'context': 2,
                      'gt_answer': 3,
                      'pred_answer': 4,
                      'overlap': 5
                      }
    labels: List[str] = list(labels_mapping.keys())
    gt_start_logits, gt_end_logits = get_start_end_arrays(gt_logits)
    pred_start_logits, pred_end_logits = get_start_end_arrays(pred_logits)
    gt_start_index = int(tf.math.argmax(gt_start_logits, axis=-1))
    gt_end_index = int(tf.math.argmax(gt_end_logits, axis=-1))
    pred_start_index = int(tf.math.argmax(pred_start_logits, axis=-1))
    pred_end_index = int(tf.math.argmax(pred_end_logits, axis=-1))
    context_start, context_end = get_context_positions(token_type_ids)
    mask[1:int(context_start - 1)] = labels_mapping['question']
    mask[int(context_start) - 1:] = labels_mapping['context']
    mask[gt_start_index:gt_end_index + 1] = labels_mapping['gt_answer']
    mask[pred_start_index:pred_end_index + 1] = labels_mapping['pred_answer']
    start_overlap = max(gt_start_index, pred_start_index)
    end_overlap = min(gt_end_index, pred_end_index)
    if start_overlap < end_overlap:
        mask[start_overlap: end_overlap + 1] = labels_mapping['overlap']
    text = get_decoded_tokens(input_ids)
    mask = mask[:len(text)]  # trancate if needed
    return mask, text, labels
