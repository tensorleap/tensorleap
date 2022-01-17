
variable "project" {
  description = "Project id where the keyring will be created."
  type        = string
}

variable "name" {
  description = "Name of the secret."
  type        = string
}

variable "data" {
  description = "Value of the secret."
  type        = string
}

