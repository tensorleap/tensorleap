# Armbench Segmentation
## General
This quick start guide will walk you through the steps to get started with this example repository project.

**Prerequisites**

Before you begin, ensure that you have the following prerequisites installed:

- **[Python](https://www.python.org/)** (version 3.7 or higher)
- **[Poetry](https://python-poetry.org/)**

### Tensorleap **CLI Installation**

with `curl`:

```
curl -s <https://raw.githubusercontent.com/tensorleap/cli-go/master/install.sh> | bash
```

with `wget`:

```
wget -q -O - <https://raw.githubusercontent.com/tensorleap/cli-go/master/install.sh> | bash
```

CLI repository: https://github.com/tensorleap/cli-go

### Tensorleap CLI Usage

#### Tensorleap **Login**
To allow connection to your Tensorleap platform via CLI you will have to authenticate and login.
To login to Tensorealp:

```
tensorleap auth login [api key] [api url].
```

- API Key is your Tensorleap token (see how to generate a CLI token in the section below).
- API URL is your Tensorleap environment URL: CLIENT_NAME.tensorleap.ai

<br> 

**How To Generate CLI Token from the UI**

1. Login to the platform in 'CLIENT_NAME.tensorleap.ai'
2. Scroll down to the bottom of the **Resources Management** page, then click `GENERATE CLI TOKEN`  in the bottom-left corner.
<img src="screenshots/Screenshot 2023-07-11 at 14.57.10.png" alt="drawing" width="1000"/>
<img src="screenshots/Screenshot 2023-07-11 at 14.57.19.png" alt="drawing" width="1000"/>
3. Once a CLI token is generated, just copy the whole text and paste it into your shell:

```
tensorleap auth login [api key] [api url]
```

### Tensorleap Dataset Deployment

To deploy your local changes:

```
tensorleap datasets save
```

#### **Tensorleap files**

Tensorleap files in the repository include `tensorleap.py` and `.tensorleap.yaml`. The files consist of the  required configurations to make the code integrate with the Tensorleap engine:

**.tensorleap.yaml**

.tensorleap.yaml file is configured to a dataset in your Tensorleap environment and is synced to the dataset saved in the environment.

For any additional file being used we add its path under `include` parameter:

```
include:
  - tensorleap.py
  - gcs_utils.py
  - [...]
```

**[tensorleap.py](http://tensorleap.py/) file**

`tensorleap.py` configure all binding functions used to bind to Tensorleap engine. These are the functions used to evaluate and train the model, visualize the variables, and enrich the analysis with external metadata variables

### Testing

To test the system we can run `test_tensorleap.py` file using poetry:

```
poetry run test
```

This file will execute several tests on [the tensorleap.py](http://tensorleap.py) script to assert that the implemented binding functions: preprocess, encoders,  metadata, etc,  run smoothly.

*For further explanation please refer to the [docs](https://docs.tensorleap.ai/)*