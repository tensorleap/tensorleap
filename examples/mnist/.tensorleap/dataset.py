import os
from typing import Union, List, Optional
from pathlib import Path
import numpy as np
from code_loader.contract.enums import DatasetInputType, DatasetOutputType, DatasetMetadataType
from code_loader.contract.datasetclasses import SubsetResponse
from code_loader import dataset_binder
from google.cloud import storage
from google.cloud.storage import Bucket
from google.auth.credentials import AnonymousCredentials
from functools import lru_cache
from tensorflow.keras.utils import to_categorical

from mnist.load_data import read_csv_file


PROJECT_ID = 'example-dev-project-nmrksf0o'
BUCKET_NAME = 'example-datasets-47ml982d'

image_size = 128

labels = np.arange(10).astype(str).tolist()


@lru_cache()
def _connect_to_gcs_and_return_bucket() -> Bucket:
    print("connect")
    # create storage client
    gcs_client = storage.Client(project=PROJECT_ID, credentials=AnonymousCredentials())
    return gcs_client.bucket(BUCKET_NAME)   # get bucket object


def _download_from_gcs(cloud_file_path: Path, local_file_path: Optional[Path] = None) -> Path:
    print("download")
    # if local_file_path is not specified saving in home dir
    if local_file_path is None:
        home_dir = os.getenv("HOME")
        local_file_path = os.path.join(home_dir, "Tensorleap_data", BUCKET_NAME, cloud_file_path)

    # check if file already exists
    if os.path.exists(local_file_path):
        return local_file_path

    bucket = _connect_to_gcs_and_return_bucket()
    dir_path = os.path.dirname(local_file_path)
    os.makedirs(dir_path, exist_ok=True)
    blob = bucket.blob(cloud_file_path)
    blob.download_to_filename(local_file_path)
    return local_file_path


def calc_classes_centroid(data: SubsetResponse):
    avg_images_dict = {}
    data_X = data['images']
    data_Y = data['labels']
    for label in labels:
        inputs_label = data_X[np.argmax(data_Y, axis=1) == label]
        avg_images_dict[label] = np.mean(inputs_label, axis=0)

    return avg_images_dict


def subset_func() -> List[SubsetResponse]:

    base_path = 'mnist/'

    cloud_file_train_path = Path(base_path, 'mnist_train.csv')
    cloud_file_test_path = Path(base_path, 'mnist_test.csv')

    local_file_train_path = _download_from_gcs(cloud_file_train_path)
    local_file_test_path = _download_from_gcs(cloud_file_test_path)

    train_X, train_Y = read_csv_file(local_file_train_path)
    test_X, test_Y = read_csv_file(local_file_test_path)

    val_split = int(len(train_X) * 0.8)
    train_X, val_X = train_X[:val_split], train_X[val_split:]
    train_Y, val_Y = train_Y[:val_split], train_Y[:val_split]

    # basic preprocessing of the input
    train_X = train_X.reshape((train_X.shape[0], 28, 28, 1))
    val_X = val_X.reshape((val_X.shape[0], 28, 28, 1))
    test_X = test_X.reshape((test_X.shape[0], 28, 28, 1))

    # one hot encode the target values
    train_Y = to_categorical(train_Y)
    val_Y = to_categorical(val_Y)
    test_Y = to_categorical(test_Y)

    train = SubsetResponse(length=len(train_X), data={'images': train_X,
                                                      'labels': train_Y,
                                                      'shape': image_size
                                                      })

    val = SubsetResponse(length=len(val_X), data={'images': val_X,
                                                  'labels': val_Y,
                                                  'shape': image_size
                                                  })

    test = SubsetResponse(length=len(val_X), data={'images': test_X,
                                                   'labels': test_Y,
                                                   'shape': image_size   # TODO: do I need it? for what purpose
                                                   })

    avg_images_dict = calc_classes_centroid(train)
    dataset_binder.cache_container["word_to_index"]["classes_avg_images"] = avg_images_dict # TODO: do we need to keep inside "word_to_index"? or as another key?

    response = [train, val, test]
    return response


def input_encoder(idx: int, subset: SubsetResponse) -> np.ndarray:
    """ preprocess the input sample """
    input = subset.data['inputs'][idx]
    return input[..., np.newaxis]


def gt_encoder(idx: int, subset: Union[SubsetResponse, list]) -> np.ndarray:
    """" preprocess the ground truth """
    label = subset.data['labels'][idx]
    return label[..., np.newaxis]


""" here we extract some metadata we want for our data """


def metadata_sample_index(idx: int, subset: Union[SubsetResponse, list]) -> np.ndarray:
    """ save the sample index number """
    return idx


def metadata_sample_average_brightness(idx: int, subset: Union[SubsetResponse, list]) -> np.ndarray: # TODO : output the most closest class per image based on eualideian dis
    """ calculate avrage pixels values per image """
    sample_input = subset.data['inputs'][idx]
    return np.mean(sample_input)


def metadata_euclidean_diff_from_class_centroid(idx: int, subset: Union[SubsetResponse, list]) -> np.ndarray:   # TODO: make sure it's possible to input all dataset
    """ calculate euclidean distance from the average image of the specific class"""
    data_X = subset.data['inputs']
    data_Y = subset.data['labels']
    sample_input = data_X[idx]
    label = np.argmax(data_Y[idx])
    inputs_label = data_X[np.argmax(data_Y, axis=1) == label]
    class_average_image = np.mean(inputs_label, axis=0)
    return np.linalg.norm(class_average_image - sample_input)


def metadata_encoder_most_similar_class(idx: int, subset: Union[SubsetResponse, list]) -> np.ndarray: # TODO : change to average color per image
    """ calculate euclidean distance from the average vector of the specific class"""
    data_X = subset.data['inputs']
    data_Y = subset.data['labels']
    sample_input = data_X[idx]
    label = np.argmax(data_Y[idx])
    inputs_label = data_X[np.argmax(data_Y, axis=1) == label]
    class_average_vec = np.mean(inputs_label, axis=0)



    return np.linalg.norm(class_average_vec - sample_input)


dataset_binder.set_subset(function=subset_func, name='images')

dataset_binder.set_input(function=input_encoder,
                         subset='images',
                         input_type=DatasetInputType.Image,
                         name='images')

dataset_binder.set_ground_truth(function=gt_encoder,
                                subset='images',
                                ground_truth_type=DatasetOutputType.Classes,
                                name='classes',
                                labels=labels)

dataset_binder.set_metadata(function=metadata_sample_index,
                            subset='images',
                            metadata_type=DatasetMetadataType.float,
                            name='sample_number')

dataset_binder.set_metadata(function=metadata_euclidean_diff_from_class_centroid, subset='images',
                            metadata_type=DatasetMetadataType.float,
                            name='euclidean_diff_from_class_centroid')
