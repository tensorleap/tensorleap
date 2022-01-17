resource "google_storage_bucket" "datasets" {
  name     = "${local.client_name}-datasets"
  project  = local.project_id
  location = "US"
  force_destroy = true

  labels = {
    terraform : true
  }
}

