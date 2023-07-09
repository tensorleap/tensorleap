import numpy as np
import numpy.typing as npt
from typing import List
import tensorflow as tf
from textblob import TextBlob

from albert.data_set import get_context_positions, max_sequence_length, PAD_TOKEN
from albert.metrices import get_start_end_arrays




def tokens_context_decoder(input_ids, token_type_ids, tokenizer):
    input_ids = input_ids.astype(np.int32).tolist()
    context_start, context_end = get_context_positions(token_type_ids)
    input_ids = input_ids[int(context_start): int(context_end + 1)]
    decoded = tokenizer.decode(input_ids, max_seq_length=max_sequence_length)
    decoded = decoded.split(' ')
    return decoded

def tokens_question_decoder(input_ids, token_type_ids, tokenizer):
    input_ids = input_ids.astype(np.int32).tolist()
    context_start, context_end = get_context_positions(token_type_ids)
    input_ids = input_ids[1:int(context_start - 1)]
    decoded = tokenizer.decode(input_ids, max_seq_length=max_sequence_length)
    decoded = decoded.split(' ')
    return decoded


def tokens_decoder(decoded):
    if len(decoded) < max_sequence_length:  # pad
        decoded += (max_sequence_length - len(decoded)) * [PAD_TOKEN]
    elif len(decoded) > max_sequence_length:  # truncate
        decoded = decoded[:max_sequence_length]
    return decoded



def tokenizer_decoder(tokenizer, input_ids):
    decoded = tokenizer.decode(input_ids)
    decoded = decoded.split(' ')
    return decoded


def answer_decoder(logits, input_ids, tokenizer):
    start_logits, end_logits = get_start_end_arrays(logits)
    input_ids = input_ids.astype(np.int32).tolist()
    start_index = int(tf.math.argmax(start_logits, axis=-1))
    end_index = int(tf.math.argmax(end_logits, axis=-1))
    selected_answer_ids = input_ids[start_index:end_index + 1]
    answer = tokenizer_decoder(tokenizer, selected_answer_ids)
    return answer


def context_subjectivity(text):
    blob = TextBlob(text)
    val = blob.subjectivity
    if val is None:
        val = -1
    return val

def context_polarity(text):
    blob = TextBlob(text)
    val = blob.polarity
    if val is None:
        val = -1
    return val



def get_decoded_tokens(input_ids, tokenizer):
    input_ids = input_ids.astype(np.int32).tolist()
    decoded = tokenizer.convert_ids_to_tokens(input_ids)
    ind = decoded.index('<pad>') if '<pad>' in decoded else None
    decoded = decoded[:ind] if ind is not None else decoded  # truncate
    decoded = [token.replace(chr(9601), '') for token in decoded]
    return decoded

def segmented_tokens_decoder(input_ids, token_type_ids, gt_logits, pred_logits):
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
