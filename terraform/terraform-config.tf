terraform {
  required_version = "~> 0.14.3"
  backend "gcs" {
    bucket = "tensorleap-infra-nonprod"
    prefix = "clients/example/"
  }
  required_providers {
    google = {
      version = "~> 3.51.0"
    }
    google-beta = {
      version = "~> 3.51.0"
    }
  }
}

