resource "google_service_account" "leap-account" {
  account_id   = "leap-account-service-account"
  display_name = "Terraform-managed."
  project      = local.project_id

}

resource "google_service_account_key" "leap-account" {
  service_account_id = google_service_account.leap-account.email
}

resource "google_storage_bucket_iam_member" "dataset-gcs" {
  bucket = local.dataset_bucket
  role   = "roles/storage.admin"
  member = "serviceAccount:${google_service_account.leap-account.email}"
}

module "service-account-key-file-secret" {
  source  = "./secret"
  project = local.project_id
  name    = local.dataset_bucket
  data    = base64decode(google_service_account_key.leap-account.private_key)
}

