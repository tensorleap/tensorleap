# RAFT optical flow

**RAFT** is an optical flow model that uses an all-pairs recurrent model to approximate the optical flow solution.
In this example, we infer [RAFT](https://github.com/princeton-vl/RAFT) on the
[KITTI](https://www.cvlibs.net/datasets/kitti/index.php) dataset .

**Prerequisites**

Before you begin, ensure that you have the following prerequisites installed:

- **[Python](https://www.python.org/)** (version 3.7 or higher)
- **[Poetry](https://python-poetry.org/)**

## Tensorleap **CLI Installation**

with `curl`:

```
curl -s <https://raw.githubusercontent.com/tensorleap/cli-go/master/install.sh> | bash
```

with `wget`:

```
wget -q -O - <https://raw.githubusercontent.com/tensorleap/cli-go/master/install.sh> | bash
```

CLI repository: https://github.com/tensorleap/cli-go

## Tensorleap CLI Usage

### Tensorleap **Login**

To login to Tensorealp:

```
tensorleap auth login [api key] [api url].

```

- API Key is your Tensorleap token (see how to generate a CLI token in the section below).
- API URL is your Tensorleap environment URL: CLIENT_NAME.tensorleap.ai

<br>

**How To Generate CLI Token from the UI**

1. Login to the platform in 'CLIENT_NAME.tensorleap.ai'
2. Scroll down to the bottom of the **Resources Management** page, then click `GENERATE CLI TOKEN` in the bottom-left corner.
3. Once a CLI token is generated, just copy the whole text and paste it into your shell:

```
tensorleap auth login [api key] [api url]

```

## Tensorleap **Dataset Deployment**

To deploy your local changes:

```
tensorleap datasets push

```

### **Tensorleap files**

Tensorleap files in the repository include `tensorleap.py` and `.tensorleap.yaml`. The files consist of the  required configurations to make the code integrate with the Tensorleap engine:

**.tensorleap.yaml**

.tensorleap.yaml file is configured to a dataset in your Tensorleap environment and is synced to the dataset saved in the environment.

For any additional file being used we add its path under `include` parameter:

```
include:
  - tensorleap.py
	- cs_data.py
  - kitti_data.py
  - configs.py
  - gcs_utils.py

```

**[tensorleap.py](http://tensorleap.py/) file**

`tensorleap.py` configure all binding functions used to bind to Tensorleap engine. These are the functions used to evaluate and train the model, visualize the variables, and enrich the analysis with external metadata variables

## Testing

To test the system we can run `[test.py](http://test.py/)` file using poetry:

```
poetry run test

```

This file will execute several tests on [the tensorleap.py](http://tensorleap.py/) script to assert that the implemented binding functions: preprocess, encoders,  metadata, etc,  run smoothly.

*For further explanation please refer to the [docs](https://docs.tensorleap.ai/)*

# Latent Space and Clusters

After inferring RAFT on two KITTI subsets (scene-flow and stereo-flow) using the Tensorleap platform we get a
visualization of our latent space.

![Untitled](screenshots/1.png)

Coloring the latent space according to our TSNE clusters we get several distinct clusters:

![Untitled](screenshots/2.png)

Going through some of the clusters we can see that we have clusters that contain:

- image pairs where the car takes a left turn:  

![Untitled](screenshots/left_turns.gif)

- image pairs where the car takes a right turn:

![Untitled](screenshots/right_turns.gif)

- image pairs where the car has no ego motion:

![Untitled](screenshots/no-ego.gif)

# Dashboards

![Untitled](screenshots/dashboard.png)

In the Dashboard panel we can see the correlation of various metadata with the loss and FL metrics:

- The focus of expnasion location vs. the loss (high error when taking turns)
- Average Optical Flow magnitude vs. loss/Fl-metric
- Subset Name vs. Loss
- Amount of max pixels vs. loss (more pixels masked - higher error)