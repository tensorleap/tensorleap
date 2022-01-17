locals {
  project_id = module.project.project_id
  client_name = "trial"
  dataset_bucket = google_storage_bucket.datasets.name
  owners = [
    "user:yotam@tensorleap.ai",
    "user:tom.koren@tensorleap.ai"
  ]
  editors = [
    "user:tom.koren@tensorleap.ai"
  ]
}
