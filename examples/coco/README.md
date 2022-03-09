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


# COCO

#### The Task

In this example, we show the use of Tensorleap on a Computer Vision task - Semantic Segmentation. The goal of semantic image segmentation is to label each pixel of an image with a corresponding class. Because the prediction is for every pixel in the image, this task is commonly referred to as dense prediction. In Semantic Segmentation, the boundary of objects are labeled with a mask and object classes are labeled with a class label. The output itself is a high resolution image (typically of the same size as input image) in which each pixel is classified to a particular class. 

#### The COCO dataset

The dataset we use is COCO dataset. COCO (Common Objects in Context) is large-scale object detection, segmentation, and captioning dataset. The COCO Dataset has 80 classes.

### The modified U-Net

#### Model introduction

We use a modified U-Net, which use a Fully Convolutional Network Model for the task. The U-Net was developed by Olaf Ronneberger et al. for Bio Medical Image Segmentation. The architecture contains two paths. First path is the contraction path (also called as the encoder) which is used to capture the context in the image. The encoder is just a traditional stack of convolutional and max pooling layers. The second path is the symmetric expanding path (also called as the decoder) which is used to enable precise localization using transposed convolutions. Thus it is an end-to-end fully convolutional network (FCN).

In the original paper, the UNET is described as follows:

![](https://hackmd.io/_uploads/SJKovyBWq.png)

#### Transfer Learning:

For the encoder, we use a pretrained model - MobileNetV2. That will help to learn robust features and reduce the number of trainable parameters. The decoder/upsampler is simply a series of upsample blocks implemented in TensorFlow examples. During training process, only the decoder will be trained when the encoder weigths are frozen.

#### Category Prediction Model

The model's task is to segment images consist of two categories: `person` and `car`.  


#### Cluster Analysis

First, we evaluate our model on a dataset subset containing cars and person instances:

<span style="color:blue">TODO add image</span>

To quantify the model's predictions on the different classes we utilize Tesnorleap's latent space. We select samples from different areas of the embedding space, and used `fetch similar` to create unique clusters of similar samples.

Among the clusters we got are:


#### B&W cluster 

We fetch group of similar images that are mostly on grey scale colors as  seen bellow:
![](https://hackmd.io/_uploads/S1LBQZU-c.png)

From further look we get that most of the samples are structured with 3 channels  (most of samples are colored as red from 'is_colored' metadata): 
![](https://hackmd.io/_uploads/SyOlVbLbc.png)

However, the system did find those grey scale look similar.

![](https://hackmd.io/_uploads/SkKKdu5e9.jpg) [remove](/QBn3a4vZR5urM2a38dwd5Q)

This cluster has a much higher loss than the other, non-grayscale, images.
<span style="color:blue">Todo: Support this with metadata </span>

### Vehicle-like clusters

The system was able to extract clusters chrachterized by categories which the net didn't train on as further seen. A reasonable explenation is that the pretrained encoder we use was trained on these classes and is extracting those features classes.  

#### Bicycle cluster 
 ![](https://hackmd.io/_uploads/rkZpLcfxq.png)

#### Bus cluster 
![](https://hackmd.io/_uploads/HJ_N-2Qb9.png)

We can see that this cluster has a high concentration of buses and cars. However, we also see buildings and poles as part of the same cluster. 

![](https://hackmd.io/_uploads/SyFIDsNbc.png)

Examining the attention map on the features that make this cluster similar, we see features that are not directly related to vehicles (towers, etc.). Also we see that the loss on images containing bus are high compared to one that does not have buses
<span style="color:blue">Todo: show this using dashboard</span>

### Vehicle Supercategory model

From the vehicle-like clusters we found we can conclude that giving the model to segment other vehicles' wheels (which aren't car) as background might negatively affect the performance. That leads us to redefine our categories classes as `vehicle` and `person`.  

Evaluating the new model on a Super Category (SC) labeled dataset we get:


| Dataset | Train | Test |
| -------- | -------- | -------- |
| Category Model     | 0.184     | 0.297     |
| Super Category Model    | 0.216     | 0.256     |

<span style="color:blue">Todo: discuss the scores</span>

Fetching similars to one of the vehicles result in a more homogenous cluster (composed of cars + buses).
<span style="color:blue">Todo: add images of cluster</span>
One of the strongest shared features in this cluster are the wheels:
![](https://hackmd.io/_uploads/BJhkzm5xq.png)

Reviewing the attention map reveals a possible confusion: round objects (camera lens) could be categorized as cars due to their similarity to wheels:
![](https://hackmd.io/_uploads/rkGf-X5l5.png)
![](https://hackmd.io/_uploads/SJEV-7cxq.png)
![](https://hackmd.io/_uploads/H1B_Zm9l5.png)
![](https://hackmd.io/_uploads/ryA5WXqxc.png)
![](https://hackmd.io/_uploads/rJF3bQ5eq.png)


#### Another Cluster: Sport cluster

![](https://hackmd.io/_uploads/HyX4YwU-9.png)




<span style="color:blue">TODO show the difference on the bus category with metrics/dashboard</span>.

![](https://hackmd.io/_uploads/SJb9arLWq.png)

The top chart is of the `car` category model and the bottom is the `vehicle` super category model. For `car` category model the average loss descreases when number of busses in image increases, as for train set and as for validation set. When there are more bus objects within the image the model is less likely to mistake on `car` category.
As for the `vehicle` super category model, the loss on `vehicle` category increases when number of bus objects increases.

<span style="color:blue">TODO explain why we improved (hopefully) prediction on buses category? explain the plot?</span>.

#### False and Ambigous Labels

<span style="color:blue">TODO using cluster + dashboard to highlight COCO/Semantic-seg issues</span>.

With the help of Sample Analysis feature, we can find some ambigous labels and false labeling images that we can exclude from our train dataset to improve the performance.

GT: Mislabeled Image: two missing people:
![](https://hackmd.io/_uploads/Sk2WrWr-c.png)

- The driver and the woman from left are segmented as car.

Prediction: The model correctly segment the all three peope.
![](https://hackmd.io/_uploads/Bkilr-Sb9.png)

- Sample error analysis revield that it wasn't the model's false prediction but rather false labeling.

Ambigous label Image: Toy car as a car :)
![](https://hackmd.io/_uploads/rJuxnD9e5.png)


Challenging image: Low light
![](https://hackmd.io/_uploads/B1BT5vqgc.png)

Challenging image: Crowds with small people
![](https://hackmd.io/_uploads/ry_IoP5x5.png)

Ambigous label Image: Windshield
![](https://hackmd.io/_uploads/rJuxnD9e5.png)



#### Other interesting metadata

![](https://hackmd.io/_uploads/H12-h1rWc.png)

We can see that when the number of person instances per image increases the Cross Entropy Loss increases and the mean IoU decreases. The net's predictions are less accurates when the image is more densed with objects. 

For the vehicle category, we can see that model performance decreases when the vehicle average size increases. 
![](https://hackmd.io/_uploads/r1R7RkHb5.png)


For all categories: total instances count:
![](https://hackmd.io/_uploads/rJElUv8b9.png)



Object Average size:


![](https://hackmd.io/_uploads/r1Azdw8Z9.png)
When the average person size per image increase the mean IoU on person decreases. The segmentation is less accurate.


#### Summary and Conclusion


<span style="color:blue">TODO</span>.
