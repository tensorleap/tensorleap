
resource "google_secret_manager_secret" "default" {
  provider  = google-beta
  project   = var.project
  secret_id = var.name
  labels = {
    terraform : true
  }

  replication {
    automatic = true
  }
}

resource "google_secret_manager_secret_version" "default" {
  provider    = google-beta
  secret      = google_secret_manager_secret.default.id
  enabled     = true
  secret_data = var.data
}

