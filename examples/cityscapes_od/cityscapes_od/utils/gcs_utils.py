from google.cloud import storage
from google.cloud.storage import Bucket
from google.oauth2 import service_account
import os
import json
from functools import lru_cache
from typing import Optional

from cityscapes_od.config import CONFIG

@lru_cache()
def _connect_to_gcs_and_return_bucket(bucket_name: str) -> Bucket:
    """
        This function establishes a connection to Google Cloud Storage (GCS) using the provided authentication credentials.
        Input: bucket_name (str): The name of the Google Cloud Storage (GCS) bucket to connect to.
        Output: Returns a Bucket object representing the specified GCS bucket.
    """
    auth_secret_string = os.environ['AUTH_SECRET']
    auth_secret = json.loads(auth_secret_string)
    if type(auth_secret) is dict:
        # getting credentials from dictionary account info
        credentials = service_account.Credentials.from_service_account_info(auth_secret)
    else:
        # getting credentials from path
        credentials = service_account.Credentials.from_service_account_file(auth_secret)
    project = credentials.project_id
    gcs_client = storage.Client(project=project, credentials=credentials)
    return gcs_client.bucket(bucket_name)


def _download(cloud_file_path: str, local_file_path: Optional[str] = None) -> str:
    """
    This function downloads a file from a specified location within a Google Cloud Storage (GCS) bucket to the local
    filesystem.
    Inputs: cloud_file_path (str): The path of the file within the GCS bucket that needs to be downloaded.
            local_file_path (Optional[str]): The path where the file should be saved on the local filesystem.
    Output: Returns the path (str) to the downloaded file on the local filesystem.
    """
    # if local_file_path is not specified saving in home dir
    if local_file_path is None:
        home_dir = os.getenv("HOME")
        local_file_path = os.path.join(home_dir, "Tensorleap", CONFIG['BUCKET_NAME'], cloud_file_path)

    # check if file already exists
    if os.path.exists(local_file_path):
        return local_file_path
    bucket = _connect_to_gcs_and_return_bucket(CONFIG['BUCKET_NAME'])
    dir_path = os.path.dirname(local_file_path)
    os.makedirs(dir_path, exist_ok=True)
    blob = bucket.blob(cloud_file_path)
    blob.download_to_filename(local_file_path)
    return local_file_path
