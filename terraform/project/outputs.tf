output "project_id" {
  description = "Project id."
  value       = google_project.project.project_id
  depends_on = [
    google_project_service.project_services
  ]
}

output "project_number" {
  description = "Project number."
  value       = google_project.project.number
  depends_on = [
    google_project_service.project_services
  ]
}

