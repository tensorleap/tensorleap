import os
from typing import Optional, List, Union, Tuple
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader import leap_binder
from google.cloud import storage
from google.cloud.storage import Bucket
from google.auth.credentials import AnonymousCredentials
from functools import lru_cache
import numpy as np
import pandas as pd
from pathlib import Path
from keras.utils.np_utils import to_categorical

from code_loader.contract.enums import DatasetMetadataType, Metric


PROJECT_ID = 'example-dev-project-nmrksf0o'
BUCKET_NAME = 'example-datasets-47ml982d'

IMAGE_DIM = 28
LABELS = np.arange(10).astype(str).tolist()


###########################
########## Utils ##########
###########################


def preprocess(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    print('preprocessing the data')
    data_X = df.drop('label', axis=1).to_numpy()
    data_X = np.expand_dims(data_X, axis=-1)  # Reshape :,28,28 -> :,28,28,1
    data_X = data_X / 255  # Normalize to [0,1]
    data_Y = df.label.to_numpy()
    # one hot encode the target values
    data_Y = to_categorical(data_Y)
    return [data_X, data_Y]


#############################
########## Helpers ##########
#############################


@lru_cache()
def _connect_to_gcs_and_return_bucket(bucket_name: str) -> Bucket:
    gcs_client = storage.Client(project=PROJECT_ID, credentials=AnonymousCredentials())
    return gcs_client.bucket(bucket_name)


def _download(cloud_file_path: str, local_file_path: Optional[str] = None) -> str:
    # if local_file_path is not specified saving in home dir
    if local_file_path is None:
        home_dir = os.getenv("HOME")
        local_file_path = os.path.join(home_dir, "Tensorleap_data", BUCKET_NAME, cloud_file_path)

    # check if file is already exists
    if os.path.exists(local_file_path):
        return local_file_path

    bucket = _connect_to_gcs_and_return_bucket(BUCKET_NAME)
    dir_path = os.path.dirname(local_file_path)
    os.makedirs(dir_path, exist_ok=True)
    blob = bucket.blob(cloud_file_path)
    blob.download_to_filename(local_file_path)
    return local_file_path


####################################
########## TL Integration ##########
####################################


def subset_func() -> List[PreprocessResponse]:

    base_path = 'mnist/'

    cloud_file_train_path = str(Path(base_path, 'mnist_train.csv'))
    cloud_file_test_path = str(Path(base_path, 'mnist_test.csv'))

    local_file_train_path = _download(cloud_file_train_path)
    local_file_test_path = _download(cloud_file_test_path)

    df = pd.read_csv(local_file_train_path)
    train_X, train_Y = preprocess(df)

    df = pd.read_csv(local_file_test_path)
    test_X, test_Y = preprocess(df)

    val_split = int(len(train_X) * 0.8)
    train_X, val_X = train_X[:val_split], train_X[val_split:]
    train_Y, val_Y = train_Y[:val_split], train_Y[val_split:]

    train = PreprocessResponse(length=len(train_X), data={'images': train_X,
                                                      'labels': train_Y
                                                      })

    val = PreprocessResponse(length=len(val_X), data={'images': val_X,
                                                  'labels': val_Y
                                                  })

    test = PreprocessResponse(length=len(test_X), data={'images': test_X,
                                                    'labels': test_Y
                                                    })

    def calc_classes_centroid(subset: PreprocessResponse) -> dict:
        """ calculate average image on the pixels.
         returns a dictionary: key: class, values: images 28x28 """
        avg_images_dict = {}
        data_X = subset.data['images']
        data_Y = subset.data['labels']
        for label in LABELS:
            inputs_label = data_X[np.equal(np.argmax(data_Y, axis=1), int(label))]
            avg_images_dict[label] = np.mean(inputs_label, axis=0)
        return avg_images_dict

    avg_images_dict = calc_classes_centroid(train)
    leap_binder.cache_container["classes_avg_images"] = avg_images_dict
    response = [train, val, test]
    return response


def input_encoder(idx: int, subset: PreprocessResponse) -> np.ndarray:
    return subset.data['images'][idx].astype('float32')


def gt_encoder(idx: int, subset: Union[PreprocessResponse, list]) -> np.ndarray:
    return subset.data['labels'][idx].astype('float32')


#############################################
########## Bind metadata Functions ##########
#############################################


def metadata_sample_index(idx: int, subset: Union[PreprocessResponse, list]) -> np.ndarray:
    """ save the sample index number """
    return idx


def metadata_label(idx: int, subset: Union[PreprocessResponse, list]) -> int:
    """ save the sample index number """
    label = gt_encoder(idx, subset)
    idx = label.argmax()
    return int(idx)


def metadata_sample_average_brightness(idx: int, subset: Union[PreprocessResponse, list]) -> np.ndarray:
    """ calculate average pixels values per image """
    sample_input = subset.data['images'][idx]
    return np.mean(sample_input)


def metadata_euclidean_diff_from_class_centroid(idx: int, subset: Union[PreprocessResponse, list]) -> np.ndarray:
    """ calculate euclidean distance from the average image of the specific class"""
    sample_input = subset.data['images'][idx]
    label = subset.data['labels'][idx]
    label = str(np.argmax(label))
    class_average_image = leap_binder.cache_container["classes_avg_images"][label]
    return np.linalg.norm(class_average_image - sample_input)


def calc_most_similar_class_not_gt_label_euclidean_diff(idx: int, subset: PreprocessResponse) -> Tuple[float, str]:
    """ find the most similar class average image (which isn't the ground truth)
    based on euclidean distance from the sample"""
    sample_input = subset.data['images'][idx]
    label = subset.data['labels'][idx]
    label = str(np.argmax(label))
    distance = 0.
    min_distance = float('+inf')
    min_distance_class_label = None
    for label_i in LABELS:
        if label_i == label:
            continue
        class_average_image = leap_binder.cache_container["classes_avg_images"][label_i]
        distance = np.linalg.norm(class_average_image - sample_input)
        if distance < min_distance:
            min_distance = distance
            min_distance_class_label = label_i
    return [min_distance, min_distance_class_label]


def metadata_most_similar_class_not_gt(idx: int, subset: Union[PreprocessResponse, list]) -> np.ndarray:
    _, min_distance_class_label = calc_most_similar_class_not_gt_label_euclidean_diff(idx, subset)
    return min_distance_class_label


def metadata_most_similar_class_not_gt_diff(idx: int, subset: Union[PreprocessResponse, list]) -> np.ndarray:
    min_distance, _ = calc_most_similar_class_not_gt_label_euclidean_diff(idx, subset)
    return min_distance


leap_binder.set_preprocess(subset_func)

leap_binder.set_input(input_encoder, 'image')

leap_binder.set_ground_truth(gt_encoder, 'classes')
leap_binder.add_prediction('classification', LABELS, [Metric.MeanSquaredError, Metric.Accuracy])

leap_binder.set_metadata(metadata_sample_index, DatasetMetadataType.int, 'sample_index')
leap_binder.set_metadata(metadata_sample_average_brightness, DatasetMetadataType.float, 'sample_average_brightness')
leap_binder.set_metadata(metadata_label, DatasetMetadataType.int, 'label')
leap_binder.set_metadata(metadata_euclidean_diff_from_class_centroid, DatasetMetadataType.float, 'euclidean_diff_from_class_centroid')
leap_binder.set_metadata(metadata_most_similar_class_not_gt, DatasetMetadataType.string, 'most_similar_class_not_gt_label')
leap_binder.set_metadata(metadata_most_similar_class_not_gt_diff, DatasetMetadataType.float, 'most_similar_class_not_gt_diff')

