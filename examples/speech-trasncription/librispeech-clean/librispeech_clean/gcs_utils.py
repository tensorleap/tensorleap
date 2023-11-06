from functools import lru_cache
from os import getenv, path, makedirs
from typing import Optional

from google.cloud import storage
from google.cloud.storage import Bucket
from google.oauth2 import service_account

from librispeech_clean.configuration import config


@lru_cache()
def _connect_to_gcs_and_return_bucket(bucket_name: str) -> Bucket:
    gcs_client = storage.Client.create_anonymous_client()
    return gcs_client.bucket(bucket_name)


def download(cloud_file_path: str, local_file_path: Optional[str] = None) -> str:
    bucket_name = config.get_parameter('bucket_name')
    # if local_file_path is not specified saving in home dir
    if local_file_path is None:
        home_dir = getenv("HOME")
        local_file_path = path.join(home_dir, "Tensorleap", bucket_name, cloud_file_path)

    # check if file already exists
    if path.exists(local_file_path):
        return local_file_path
    bucket = _connect_to_gcs_and_return_bucket(bucket_name)
    dir_path = path.dirname(local_file_path)
    makedirs(dir_path, exist_ok=True)
    blob = bucket.blob(cloud_file_path)
    blob.download_to_filename(local_file_path)
    return local_file_path
