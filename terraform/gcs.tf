resource "google_storage_bucket" "datasets" {
  name     = "${local.client_name}-datasets"
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