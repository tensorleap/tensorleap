# README

This is a base repository for Tensorleap's projects.  
It contains two elements:
1) The .tensorleap folder containing the code needed to integrate a model and a dataset with the Tensorleap's system
2) Terraform files required to open a GCP project to host the dataset
## Setting up environment
Follow these steps to set up the repository locally:

### Using poetry
```shell=
brew install poetry
brew install pyenv

pyenv install 3.6.9
pyenv local use 3.6.9

poetry env use $( pyenv which python )
poetry install

```

## Storing the data
In order to evaluate and train on the data we need to store the data.
Under the `./terraform` folder we hold a terraform configuration that:

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
## Tensorleap Usage 

### Setting up the project and the dataset

1. Login to CLIENT_NAME-webapp2.tensorleap.ai
2. Create a new project
3. Save the secret created by terraform locally:  
``gcloud secrets versions --project CLIENT-NAME-dev-project access 1 --secret="CLIENT-NAME-datasets" > secret.json``
4. Go to resource management and Click "New Secret"
5. Add the json file, name the secret, and save
6. Create a new dataset:
   - Name it
   - Use the stored secret to give it access rights to the GCP bucket
   - Fill the bucket name with: `CLIENT_NAME-datasets`
   - Save the dataset

### Filling in python files

First we need to fill in the required configuration and files to make the code integrate with the tensorleap engine:
1. In `config.toml` configure the metadata variables
2. In `dataset.py` configure the data sources used to evaluate and train the model
3. In `model.py` configure the model integrated with Tensorleap's system

### Testing and deployment
In order to test/deploy we first need to activate the virtual environment used by poetry by running:
`poetry shell`
#### Testing
To test the system we can run: `leap check`.
We need to treat any errors, until the test concludes with an "OK" status.  
#### Deployment
After the tests pass, we use `leap push` to push the code into the repository. 
