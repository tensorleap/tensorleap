

## Project Description
### Resnet18 model with CIFAR-10 dataset

This project utilizes the Resnet18 algorithm to perform image classification on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. The 
CIFAR-10 dataset comprises 60,000 32x32 color images distributed across 10 classes, with each class containing 6,000 
images. The ground truth labels for the classes are as follows: 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 
'frog', 'horse', 'ship', and 'truck'.

### Latent Space Exploration

The following plot illustrates a population exploration map, depicting the similarity among samples based on the latent 
space of a trained model. This map is constructed using the extracted features of the model.

The visualization show the all the data points, revealing two distinct clusters. These clusters represent images labeled 
as animals or vehicles according to their ground truth labels. The noticeable separation between the clusters indicates 
a significant difference between these two categories.

![distinct_clusters_animal_vehicles](images/distinct_clusters_animal_vehicles.png)

#### *Detecting & Handling High Loss Clusters*

After conducting further analysis, it has been observed that the 'cat' cluster consists of several samples with higher 
loss, indicated by larger dot sizes on the plot. A closer examination reveals that a significant number of these samples 
were incorrectly predicted as 'dog' by the model. This suggests that there is a need to include more images of cats and 
possibly dogs in the training dataset to improve the model's ability to recognize them accurately.

![high_loss_cat](images/Screenshot 2023-07-13 at 11.27.27.png)


#### *Detecting & Handling High Loss Unlabeled Clusters* 

In the k-means clusters, clusters 5 and 2 are observed to be in proximity to each other. Cluster 5 predominantly 
consists of images with a light background and objects exhibiting an orange shade. On the other hand, cluster number 2 
also has a light background, but the objects within it appear in a darker shade.

##### *<u>- cluster 5: </u>* 
<div style="display: flex">
  <img src="images/cluster_2_5_kmeans/cluster_5_1.png" alt="Image 2" style="margin-right: 10px;">
  <img src="images/cluster_2_5_kmeans/cluster_5_2.png" alt="Image 3" style="margin-left: 10px;">
</div> 

##### *<u>- cluster 2: </u>* 

<div style="display: flex">
  <img src="images/cluster_2_5_kmeans/cluster_2_1.png" alt="Image 4" style="margin-right: 10px;">
  <img src="images/cluster_2_5_kmeans/cluster_2_2.png" alt="Image 5" style="margin-left: 10px;">
</div>


##### fetching similar

An alternative method for identifying clusters in the model's latent space is to retrieve similar samples based on a 
selected sample. This approach allows you to pinpoint a cluster that exhibits a specific intrinsic property of interest. 
The figure presented below illustrates such a cluster, comprising images that feature animal faces captured from a 
profile perspective.

![image_1](images/cluset_bisecting_number_12/Screenshot 2023-07-13 at 11.04.56.png)
![image_2](images/cluset_bisecting_number_12/Screenshot 2023-07-13 at 11.05.09.png)
![image_3](images/cluset_bisecting_number_12/Screenshot 2023-07-13 at 11.05.23.png)

### Sample Loss Analysis

Within this section, we delve into the examination of features that impact the model's predictions. 
Tensorleap automatically generates a heatmap that quantifies the effect different features has on the loss. 

In the image depicted below, it is evident that the presence of background features leads to a higher loss function. 
Conversely, the inclusion of features in the nose area contributes to a lower loss function, which signifies the image's 
classification as a cat.

![cat_loss_image](images/cat_loss/image.png)
![cat_loss_background](images/cat_loss/loss_background.png)
![cat_loss_nose](images/cat_loss/loss_nose.png)


## General
This quick start guide will walk you through the steps to get started with this example repository project.

**Prerequisites**

Before you begin, ensure that you have the following prerequisites installed:

- **[Python](https://www.python.org/)** (version 3.8 or higher)
- **[Poetry](https://python-poetry.org/)**

### Tensorleap **CLI Installation**

with `curl`:

```
curl -s <https://raw.githubusercontent.com/tensorleap/leap-cli/master/install.sh> | bash
```

with `wget`:

```
wget -q -O - <https://raw.githubusercontent.com/tensorleap/leap-cli/master/install.sh> | bash
```

CLI repository: https://github.com/tensorleap/leap-cli

### Tensorleap CLI Usage

#### Tensorleap **Login**
To allow connection to your Tensorleap platform via CLI you will have to authenticate and login.
To login to Tensorealp:

```
leap auth login [api key] [api url].
```

- API Key is your Tensorleap token (see how to generate a CLI token in the section below).
- API URL is your Tensorleap environment URL: CLIENT_NAME.tensorleap.ai

<br> 

**How To Generate CLI Token from the UI**

1. Login to the platform in 'CLIENT_NAME.tensorleap.ai'
2. Scroll down to the bottom of the **Resources Management** page, then click `GENERATE CLI TOKEN`  in the bottom-left corner.
3. Once a CLI token is generated, just copy the whole text and paste it into your shell:

```
leap auth login [api key] [api url]
```

### Tensorleap Dataset Deployment

To deploy your local changes:

```
leap code push
```

### **Tensorleap files**

Tensorleap files in the repository include `leap_binder.py` and `leap.yaml`. The files consist of the  required configurations to make the code integrate with the Tensorleap engine:

### **leap.yaml file**
leap.yaml file is configured to a dataset in your Tensorleap environment and is synced to the dataset saved in the environment.

For any additional file being used we add its path under `include` parameter:

```
include:
    - leap_binder.py
    - cifar10_resnet/data/preprocessing.py
    - cifar10_resnet/config.py
    - cifar10_resnet/encoders.py
    - cifar10_resnet/project_config.yaml
    - cifar10_resnet/utils.py
```

### **leap_binder.py file**
`leap_binder.py` configure all binding functions used to bind to Tensorleap engine. These are the functions used to evaluate and train the model, visualize the variables, and enrich the analysis with external metadata variables

### Testing

To test the system we can run `leap_test.py` file using poetry:

```
poetry run test
```

This file will execute several tests on the `leap_binder.py` script to assert that the implemented binding functions: preprocess, encoders,  metadata, etc,  run smoothly.

*For further explanation please refer to the [docs](https://docs.tensorleap.ai/)*










