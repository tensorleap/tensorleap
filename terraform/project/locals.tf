locals {
  org_id          = "219080304146"
  billing_account = "01A513-C44F6C-A1B22F"
  owners = var.owners
  editors = var.editors
  services = [
    "container.googleapis.com",
    "resourceviews.googleapis.com",
    "stackdriver.googleapis.com",
    "secretmanager.googleapis.com"
  ]
}

