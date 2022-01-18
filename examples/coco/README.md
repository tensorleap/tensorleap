# README

This describes the coco integration with TL. We use `coco 14` training and validation files combined with a `monilenet_v2` backbone and a `pix2pix` based decoder. 

## Implementation details

Several adaptations had to be done to fit the network to the Tensorleap's platform:
- We've replaced  TransposeConv2D with Upsample + Conv2D for pix2pix
- We've replaced ZeroPadding2D with a paddding="same" in the following Conv2D layer

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

## Tensorleap Usage 

### Setting up the project and the dataset

1. Login to trial-webapp2.tensorleap.ai
2. Create a new project
6. Create a new dataset:
   - Name it
   - Fill the bucket name with: `example-datasets`
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
