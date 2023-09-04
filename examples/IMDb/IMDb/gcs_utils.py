from google.auth.credentials import AnonymousCredentials
from google.cloud import storage
from google.cloud.storage import Bucket
import os
from os.path import join
from typing import Optional

from IMDb.config import CONFIG
def _connect_to_gcs() -> Bucket:
    """
    Establishes a connection to Google Cloud Storage and returns a bucket object.
    Return: A GCS bucket object.
    """
    gcs_client = storage.Client(project=CONFIG['PROJECT_ID'], credentials=AnonymousCredentials())
    return gcs_client.bucket(CONFIG['BUCKET_NAME'])


def _download(cloud_file_path: str, local_file_path: Optional[str] = None) -> str:
    """
    Downloads a file from Google Cloud Storage to a local directory, ensuring it exists locally for further use.
    :param cloud_file_path: The path to the file in Google Cloud Storage.
    :param local_file_path:  The optional local path to save the downloaded file.
    :return:The path to the downloaded local file.
    """
    BASE_PATH = "imdb"
    cloud_file_path = join(BASE_PATH, cloud_file_path)
    # if local_file_path is not specified saving in home dir
    if local_file_path is None:
        home_dir = os.getenv("HOME")
        assert home_dir is not None
        local_file_path = os.path.join(home_dir, "Tensorleap_data", CONFIG['BUCKET_NAME'], cloud_file_path)
    # check if file already exists
    if os.path.exists(local_file_path):
        return local_file_path

    bucket = _connect_to_gcs()
    dir_path = os.path.dirname(local_file_path)
    os.makedirs(dir_path, exist_ok=True)
    blob = bucket.blob(cloud_file_path)
    blob.download_to_filename(local_file_path)
    return local_file_path