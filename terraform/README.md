# README

This folder contains Tensorleap's terraform files required to open a GCP project and host data.

## Storing the data
In order to evaluate and train on the data we need to store the data.
Under this folder we hold a terraform configuration that:

1. Create a new GCP project
2. Create a bucket in this GCP project
3. Create a Service acount with premision to the GCS bucket for TL platform


### Customizing the terraform files

1. Fill `locals.tf` with the name of the client and the owner of the repo
2. Go to `terraform-config.tf` and enter the name of the client
3. run the following:
```shell=
terraform plan
terraform apply
```