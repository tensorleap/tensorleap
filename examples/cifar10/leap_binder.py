from typing import List, Dict, Union
import numpy as np
import numpy.typing as npt
from keras.datasets import cifar10

# Tensorleap imports
from code_loader import leap_binder
from code_loader.contract.enums import LeapDataType
from code_loader.contract.visualizer_classes import LeapHorizontalBar
from code_loader.contract.datasetclasses import PreprocessResponse


from utils.utils import metadata_animal, metadata_fly, metadata_label_name, \
    metadata_gt_label, preprocess_func
from utils.encoders import input_encoder
from config import CONFIG


# Preprocess Function
def preprocess_func_leap() -> List[PreprocessResponse]:
    train_X, val_X, train_Y, val_Y =preprocess_func()

    # Generate a PreprocessResponse for each data slice, to later be read by the encoders.
    # The length of each data slice is provided, along with the data dictionary.
    train = PreprocessResponse(length=2000, data={'images': train_X, 'labels': train_Y, 'subset_name': 'train'})
    val = PreprocessResponse(length=1000, data={'images': val_X, 'labels': val_Y, 'subset_name': 'val'})
    response = [train, val]
    return response


def unlabeled_data() -> PreprocessResponse:
    _, (test_X, _) = cifar10.load_data()
    return PreprocessResponse(length=1000, data={'images': test_X, 'subset_name': 'unlabeled'})

# Input encoder fetches the image with the index `idx` from the `images` array set in
# the PreprocessResponse data. Returns a numpy array containing the sample's image.
def input_encoder_leap(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    # return preprocess.data['images'][idx].astype('float32')
    image = preprocess.data['images'][idx].astype('float32')
    return input_encoder(image)

# Ground truth encoder fetches the label with the index `idx` from the `labels` array set in
# the PreprocessResponse's data. Returns a numpy array containing a hot vector label correlated with the sample.
def gt_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
        return preprocess.data['labels'][idx].astype('float32')

def metadata_sample_index(idx: int, preprocess: PreprocessResponse) -> int:
    return idx

def metadata_gt_label_leap(idx: int, preprocess: PreprocessResponse) -> int:
    if preprocess.data['subset_name'] == 'unlabeled':
        return -1
    else:
        one_hot_digit = gt_encoder(idx, preprocess)
        label = metadata_gt_label(one_hot_digit)
        return label

# This metadata adds the int gt_name of each sample.
def metadata_label_name_leap(idx: int, preprocess: PreprocessResponse) -> str:
    if preprocess.data['subset_name'] == 'unlabeled':
        return 'Unlabeled'
    else:
        one_hot_digit = gt_encoder(idx, preprocess)
        label = metadata_label_name(one_hot_digit)
        return label
def metadata_fly_leap(idx: int, preprocess: PreprocessResponse) -> str:
    if preprocess.data['subset_name'] == 'unlabeled':
        return 'Unlabeled'
    else:
        one_hot_digit = gt_encoder(idx, preprocess)
        label = metadata_fly(one_hot_digit)
        return label

def metadata_animal_leap(idx: int, preprocess: PreprocessResponse) -> str:
    if preprocess.data['subset_name'] == 'unlabeled':
        return 'Unlabeled'
    else:
        one_hot_digit = gt_encoder(idx, preprocess)
        label = metadata_animal(one_hot_digit)
        return label

def horizontal_bar_visualizer_with_labels_name(data: npt.NDArray[np.float32]) -> LeapHorizontalBar:
    labels_names = [CONFIG['LABELS_NAMES'][index] for index in range(data.shape[-1])]
    return LeapHorizontalBar(data, labels_names)

def metadata_dict(idx: int, data: PreprocessResponse) -> Dict[str, Union[float, int, str]]:
    metadata_functions = {
        "sample_index": metadata_sample_index,
        "gt_label": metadata_gt_label_leap,
        "gt_label_name": metadata_label_name_leap,
        "fly": metadata_fly_leap,
        "animal": metadata_animal_leap
    }
    res = dict()
    for func_name, func in metadata_functions.items():
        res[func_name] = func(idx, data)
    return res

# Dataset binding functions to bind the functions above to the `Dataset Instance`.
leap_binder.set_preprocess(function=preprocess_func_leap)
leap_binder.set_unlabeled_data_preprocess(function=unlabeled_data)
leap_binder.set_input(function=input_encoder_leap, name='image')
leap_binder.set_ground_truth(function=gt_encoder, name='classes')
leap_binder.set_metadata(metadata_dict, name='metadata')
leap_binder.set_visualizer(horizontal_bar_visualizer_with_labels_name, 'horizontal_bar_lm', LeapDataType.HorizontalBar)
leap_binder.add_prediction(name='classes', labels=CONFIG['LABELS_NAMES'])
