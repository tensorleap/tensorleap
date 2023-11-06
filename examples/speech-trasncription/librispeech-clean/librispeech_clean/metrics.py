import numpy as np
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.keras.engine.keras_tensor import KerasTensor

from librispeech_clean.configuration import config
from librispeech_clean.utils import remove_trailing_zeros
from librispeech_clean.wav2vec_processor import ProcessorSingleton
import tensorflow as tf
from jiwer import process_words, process_characters


def ctc_loss(logits: KerasTensor, numeric_labels: EagerTensor) -> EagerTensor:
    """
    Calculate the Connectionist Temporal Classification (CTC) loss for a given set of logits and numeric labels.

    This function computes the CTC loss, which is commonly used in tasks like speech recognition and handwriting recognition
    to train models to align input sequences with their corresponding target labels.

    Args:
        logits: Logits tensor representing the predicted probabilities for each class at each time step.
        numeric_labels: Numeric labels tensor containing the target labels for alignment.

    Returns:
        A tensor representing the computed CTC loss.

    Note:
        This function will remove any trailing zeros of numeric_labels.
        that the output_length parameter must be specified correctly using `config.get_parameter('output_length')`.
    """

    numeric_labels = remove_trailing_zeros(numeric_labels.numpy()[0])
    output_length = config.get_parameter('output_length')

    # Create the labels tensor
    numeric_labels = tf.convert_to_tensor(numeric_labels, dtype=tf.int64)
    numeric_labels = tf.expand_dims(numeric_labels, axis=0)  # Adding batch dimension

    # Create logits tensors
    logits = tf.cast(logits, dtype=tf.float32)
    logits = tf.transpose(logits, perm=[0, 2, 1])

    # Convert required lengths to constants
    label_length = tf.constant([numeric_labels.shape[1]], dtype=tf.int64)
    logit_length = tf.constant([output_length], dtype=tf.int64)

    # Calculate CTC loss
    ctc_loss = tf.nn.ctc_loss(labels=numeric_labels,
                              logits=logits,
                              label_length=label_length,
                              logit_length=logit_length,
                              blank_index=0,
                              logits_time_major=False)

    return ctc_loss


def calculate_error_rate_metrics(prediction: np.ndarray, numeric_labels: np.ndarray):
    """
    Calculate error rate metrics for a given prediction and numeric labels.

    This function computes various error rate metrics such as Word Error Rate (WER), Character Error Rate (CER),
    Match Error Rate (MER), Word Information Lost (WIL), Word Information Preserved (WIP), Word Deletion, Word Insertion,
    Word Substitution, Character Deletion, Character Insertion, and Character Substitution for a given prediction and
    numeric labels. Trailing zeros in the numeric labels are automatically removed before processing.

    Args:
        prediction: A NumPy array representing the model's prediction, typically containing probabilities or logits.
        numeric_labels: A NumPy array containing the numeric target labels.

    Returns:
        A dictionary containing the computed error rate metrics as TensorFlow tensors. The dictionary includes the
        following metrics:
        - 'word_error_rate': Word Error Rate (WER)
        - 'character_error_rate': Character Error Rate (CER)
        - 'match_error_rate': Match Error Rate (MER)
        - 'word_information_lost': Word Information Lost (WIL)
        - 'word_information_preserved': Word Information Preserved (WIP)
        - 'word_deletion': Number of word deletions
        - 'word_insertion': Number of word insertions
        - 'word_substitution': Number of word substitutions
        - 'char_deletion': Number of character deletions
        - 'char_insertion': Number of character insertions
        - 'char_substitution': Number of character substitutions

    Note:
        This function automatically removes trailing zeros from the numeric_labels before computing the error rate metrics.
        It also assumes that the necessary processing functions (e.g., `ProcessorSingleton().get_processor`,
        `process_characters`, and `process_words`) are available and correctly configured in the environment.
    """

    numeric_labels = remove_trailing_zeros(numeric_labels[0])
    processor = ProcessorSingleton().get_processor()
    processed_pred = np.argmax(prediction, 2)[0]
    transcription = processor.decode(processed_pred)
    reference = processor.tokenizer.decode(numeric_labels)

    character_process_out = process_characters(reference, transcription)
    word_process_out = process_words(reference, transcription)
    wer_out = word_process_out.wer
    cer_out = character_process_out.cer
    wil_out = word_process_out.wil
    wip_out = word_process_out.wip
    mer_out = word_process_out.mer

    results = {
        'word_error_rate': wer_out,
        'character_error_rate': cer_out,
        'match_error_rate': mer_out,
        'word_information_lost': wil_out,
        'word_information_preserved ': wip_out,
        'word_deletion': word_process_out.deletions,
        'word_insertion': word_process_out.insertions,
        'word_substitution': word_process_out.substitutions,
        'char_deletion': character_process_out.deletions,
        'char_insertion': character_process_out.insertions,
        'char_substitution': character_process_out.substitutions,

    }
    results = {k: tf.convert_to_tensor([v]) for k, v in results.items()}
    return results
