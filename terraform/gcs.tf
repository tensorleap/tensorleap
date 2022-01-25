resource "google_storage_bucket" "datasets" {
  name     = "${local.client_name}-datasets-${random_string.bucket_suffix.result}"
  project  = local.project_id
  location = "US"
  force_destroy = true

  labels = {
    terraform : true
  }
}

resource "google_storage_bucket_access_control" "public_rule" {
  bucket = local.dataset_bucket
  role   = "READER"
  entity = "allUsers"
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