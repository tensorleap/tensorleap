resource "google_project" "project" {
  org_id              = local.org_id
  project_id          = var.name
  name                = var.name
  billing_account     = local.billing_account
  auto_create_network = false

  labels = {
    terraform : true
  }
}

resource "google_project_service" "project_services" {
  for_each                   = toset(local.services)
  project                    = google_project.project.project_id
  service                    = each.value
  disable_on_destroy         = true
  disable_dependent_services = true
}

resource "google_project_iam_member" "member" {
  for_each = toset(local.editors)
  project  = google_project.project.project_id
  role     = "roles/editor"
  member   = each.value
}

resource "google_project_iam_binding" "editor" {
  project = google_project.project.project_id
  role    = "roles/editor"
  members = local.editors

  depends_on = [
    google_project_service.project_services,
  ]
}

resource "google_project_iam_binding" "owners" {
  project = google_project.project.project_id
  role    = "roles/owner"
  members = local.owners

  depends_on = [
    google_project_service.project_services,
  ]
}


