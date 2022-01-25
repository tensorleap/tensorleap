resource "google_storage_bucket" "datasets" {
  name     = "${local.client_name}-datasets-${random_string.bucket_suffix.result}"
  project  = local.project_id
  location = "US"
  force_destroy = true

  labels = {
    terraform : true
  }
}


resource "random_string" "bucket_suffix" {
  length           = 8
  upper            = false
  special          = false
  override_special = "/@£$"

  lifecycle {
    ignore_changes = all
  }
}

resource "google_storage_bucket_iam_member" "dataset-gcs" {
  bucket = local.dataset_bucket
  role   = "roles/storage.objectViewer"
  member = "allUsers"
}

