> **Note**: The dataset used in this project is not public, access to it can be granted by request.
> See this link: [Cityscapes dataset](https://www.cityscapes-dataset.com/).
<be>

# YOLOv7 model with Cityscapes dataset

This project utilizes the [YOLOv7](https://github.com/WongKinYiu/yolov7) model to perform object detection on 
[Cityscapes](https://www.cityscapes-dataset.com/) dataset. The cityscapes dataset is a large-scale dataset that stands as one
of the standard ADAS benchmarks used for multiple vision-related tasks.

**Tensorleap** helps to explore the latent space of a dataset and find new and important insights. It can also be used
to find unlabeled clusters or miss-labeling in the dataset in the most efficient way.


### Latent Space Exploration
The following plot illustrates a population exploration map, depicting the similarity among samples based on the latent 
space of a trained model. This map is constructed using the extracted features of the model.

#### *Detecting & Handling High Loss Clusters*
Upon conducting additional analysis, it was found that when the number of small bounding boxes decreases, 
the object detection loss increases.

<div style="display: flex">
  <img src="images/high_loss_small_bboxes.png" alt="Image 1" style="margin-right: 10px;">
  <img src="images/img.png" alt="Image 2" style="margin-left: 10px;">
</div>

#### *Mislabeled data*

Upon conducting further analysis, it has been discovered that images containing cars exhibit a high loss. Subsequent 
research has revealed that the algorithm is actually recognizing cars accurately, but the ground truth (gt) labels are 
incorrect. The cars in the images are mislabeled. To address this issue, it is essential to correct the labels for the 
cars in order to improve the performance of the algorithm.

<div style="display: flex">
  <img src="images/bb_gt_car.png" alt="Image 3" style="margin-right: 10px;">
  <img src="images/bb_car.png" alt="Image 4" style="margin-left: 10px;">
</div>

We can observe that the class "motorcycle" is underrepresented in the data. The majority of images have zero labeled 
ground truth bounding boxes for motorcycles. To address this, we should investigate if there is any mislabeling or 
consider adding more images containing motorcycles to improve the representation of this class in the dataset.

![Unlabeled_motorcycle](images/Unlabeled_motorcycle.png)


#### *Detecting Unlabeled Clusters*

Cluster number 14 of k-means clusters characterized in dark ambiance images, featuring a road surrounded by vegetation, 
while cars are the primary objects present in the image.

![cluster14_kmeans](images/cluster14_kmeans.png)

### Sample Loss Analysis

Within this section, we delve into the examination of features that impact the model's predictions. 
Tensorleap automatically generates a heatmap that quantifies the effect different features have on loss/ specific prediction.
In the image depicted below, the heat map displays warmer areas, indicating regions in the image where features are more
prominent, and thereby increasing the likelihood that the object in those areas is a building.

![heatmap_image](images/heatmap.png)

### IOU metric
After implementing the Intersection over Union (IoU) metric for each category's class_id, it becomes evident that the 
IoU scores for the 'road' category are consistently high across most of the images. This suggests that the model's 
performance in detecting and accurately segmenting the 'road' class is notably successful in many instances.

![iou_road](images/iou_road.png)

# Getting Started with Tensorleap Project

This quick start guide will walk you through the steps to get started with this example repository project.

## Prerequisites

Before you begin, ensure that you have the following prerequisites installed:

- **[Python](https://www.python.org/)** (version 3.7 or higher).
- **[Poetry](https://python-poetry.org/)**.
- **[Tensorleap](https://tensorleap.ai/)** platform access. To request a free trial click [here](https://meetings.hubspot.com/esmus/free-trial).
- **[Tensorleap CLI](https://github.com/tensorleap/leap-cli)**.


## Tensorleap **CLI Installation**

with `curl`:

```
curl -s https://raw.githubusercontent.com/tensorleap/leap-cli/master/install.sh | bash
```

## Tensorleap CLI Usage

### Tensorleap **Login**

To login to Tensorleap:

```
tensorleap auth login [api key] [api url].
```

- API Key is your Tensorleap token (see how to generate a CLI token in the section below).
- API URL is your Tensorleap environment URL: https://api.CLIENT_NAME.tensorleap.ai/api/v2

<br>

**How To Generate CLI Token from the UI**

1. Login to the platform in 'CLIENT_NAME.tensorleap.ai'
2. Scroll down to the bottom of the **Resources Management** page, then click `GENERATE CLI TOKEN` in the bottom-left corner.
3. Once a CLI token is generated, just copy the whole text and paste it into your shell.


## Tensorleap **Project Deployment**

To deploy your local changes:

```
leap project push
```

### **Tensorleap files**

Tensorleap files in the repository include `leap_binder.py` and `leap.yaml`. The files consist of the  required configurations to make the code integrate with the Tensorleap engine:

**leap.yaml**

leap.yaml file is configured to a dataset in your Tensorleap environment and is synced to the dataset saved in the environment.

For any additional file being used, we add its path under `include` parameter:

```
include:
    - leap_binder.py
    - cityscapes_od/configs.py
    - [...]
```

**leap_binder.py file**

`leap_binder.py` configures all binding functions used to bind to Tensorleap engine. These are the functions used to evaluate and train the model, visualize the variables, and enrich the analysis with external metadata variables

## Testing

To test the system we can run `leap_test.py` file using poetry:

```
poetry run test
```

This file will execute several tests on leap_binder.py script to assert that the implemented binding functions: preprocess, encoders,  metadata, etc.,  run smoothly.

*For further explanation please refer to the [docs](https://docs.tensorleap.ai/)*



