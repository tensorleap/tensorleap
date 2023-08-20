from typing import List, Union

import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from mnist.config import CONFIG
# Tensorleap imports
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse 
from code_loader.contract.enums import Metric, DatasetMetadataType
from code_loader.contract.visualizer_classes import LeapHorizontalBar

# Preprocess Function
def preprocess_func() -> List[PreprocessResponse]:
    (train_X, train_Y), (val_X, val_Y) = mnist.load_data()

    train_X = np.expand_dims(train_X, axis=-1)  # Reshape :,28,28 -> :,28,28,1
    train_X = train_X / 255                       # Normalize to [0,1]
    train_Y = to_categorical(train_Y)           # Hot Vector
    
    val_X = np.expand_dims(val_X, axis=-1)  # Reshape :,28,28 -> :,28,28,1
    val_X = val_X / 255                     # Normalize to [0,1]
    val_Y = to_categorical(val_Y)           # Hot Vector

    # Generate a PreprocessResponse for each data slice, to later be read by the encoders.
    # The length of each data slice is provided, along with the data dictionary.
    # In this example we pass `images` and `labels` that later are encoded into the inputs and outputs 
    train = PreprocessResponse(length=len(train_X), data={'images': train_X, 'labels': train_Y})
    val = PreprocessResponse(length=len(val_X), data={'images': val_X, 'labels': val_Y})
    response = [train, val]
    return response

# Input encoder fetches the image with the index `idx` from the `images` array set in
# the PreprocessResponse data. Returns a numpy array containing the sample's image. 
def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    return preprocess.data['images'][idx].astype('float32')

# Ground truth encoder fetches the label with the index `idx` from the `labels` array set in
# the PreprocessResponse's data. Returns a numpy array containing a hot vector label correlated with the sample.
def gt_encoder(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    return preprocessing.data['labels'][idx].astype('float32')

# Metadata functions allow to add extra data for a later use in analysis.
# This metadata adds the int digit of each sample (not a hot vector).
def metadata_sample_index(idx: int, preprocess: PreprocessResponse) -> int:
    return idx

def metadata_label(idx: int, preprocess: PreprocessResponse) -> int:
    one_hot_digit = gt_encoder(idx, preprocess)
    digit = one_hot_digit.argmax()
    digit_int = int(digit)
    return digit_int

def metadata_label_name(idx: int, preprocess: PreprocessResponse) -> str:
    one_hot_digit = gt_encoder(idx, preprocess)
    digit = one_hot_digit.argmax()
    digit_int = int(digit)
    return CONFIG['LABELS_NAMES'][digit_int]

def metadata_even_odd(idx: int, preprocess: PreprocessResponse) -> str:
    one_hot_digit = gt_encoder(idx, preprocess)
    digit = one_hot_digit.argmax()
    digit_int = int(digit)
    if digit_int % 2 == 0:
        return "even"
    else:
        return "odd"

def metadata_circle(idx: int, preprocess: PreprocessResponse) -> str:
    one_hot_digit = gt_encoder(idx, preprocess)
    digit = one_hot_digit.argmax()
    digit_int = int(digit)
    if digit_int in [0, 6, 8,9]:
        return 'yes'
    else:
        return 'no'


def bar_visualizer(data: np.ndarray) -> LeapHorizontalBar:
    return LeapHorizontalBar(data, CONFIG['LABELS'])

def horizontal_bar_visualizer_with_labels_name(data: np.ndarray) -> LeapHorizontalBar:
    labels_names = [CONFIG['LABELS_NAMES'][index] for index in range(data.shape[-1])]
    return LeapHorizontalBar(data, labels_names)

# Dataset binding functions to bind the functions above to the `Dataset Instance`.
leap_binder.set_preprocess(function=preprocess_func)
leap_binder.set_input(function=input_encoder, name='image')
leap_binder.set_ground_truth(function=gt_encoder, name='classes')
leap_binder.set_metadata(function=metadata_sample_index, name='metadata_sample_index')
leap_binder.set_metadata(function=metadata_label, name='metadata_label')
leap_binder.set_metadata(function=metadata_label_name, name='metadata_label_name')
leap_binder.set_metadata(function=metadata_even_odd, name='metadata_even_odd')
leap_binder.set_metadata(function=metadata_circle, name='metadata_circle')
leap_binder.add_prediction(name='classes', labels=CONFIG['LABELS'])
leap_binder.set_visualizer(name='horizontal_bar_classes', function=bar_visualizer, visualizer_type=LeapHorizontalBar.type)
leap_binder.set_visualizer(name='horizontal_bar_classes_names', function=horizontal_bar_visualizer_with_labels_name, visualizer_type=LeapHorizontalBar.type)

