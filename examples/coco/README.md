
# README

## Semantic Segmantation - COCO 


In this example, we show the use of Tensorleap on a Computer Vision task - Semantic Segmentation on COCO data. We use `coco 14` training and validation data files combined with a `mobilenet_v2` backbone and a `pix2pix` based decoder. 

### The Task

The goal of Semantic Segmentation is to label each pixel of an image with a corresponding class. Because the prediction is for every pixel in the image, this task is commonly referred to as dense prediction. In Semantic Segmentation, the boundary of objects is labeled with a mask, and object classes are labeled with a class label. The output itself is a high-resolution image (typically of the same size as the input image) in which each pixel is classified to a particular class. 

### The Dataset

COCO (Common Objects in Context) is large-scale object detection, segmentation, and captioning dataset. The COCO Dataset has 80 classes.

### The Model

#### Model Introduction

We use a modified U-Net, which is an end-to-end Fully Convolutional Network Model for the task. The U-Net was developed by Olaf Ronneberger et al. for Biomedical Image Segmentation. The architecture consists of an encoder - which is used to capture the image context, and a decoder - which is used to enable precise localization using transposed convolutions. 

In the original paper, the U-Net is described as follows:


![Unet](./coco/images/Unet.png)

#### Transfer Learning

For the encoder, we use a pre-trained model - MobileNetV2. That will help to learn robust features and reduce the number of trainable parameters. The decoder/upsampler is simply a series of upsample blocks. During the training process, only the decoder is being trained when the encoder weights are being frozen.

## Category Prediction Model

The model's task is to segment images consisting of two categories: `person` and `car`.  

First, we evaluate our model on a dataset subset containing cars and person instances:

| Dataset | Mean IoU Person |
| -------- |  -------- |
| Category Model     | 0.309


#### Cluster Analysis

To qualitatively analyze the model's predictions on the different classes we utilize Tesnorleap's latent space. We select samples from different areas of the embedding space and use `fetch similar` to create unique clusters of similar samples.

Among the clusters we got are:

#### B&W cluster 

We fetch a group of similar images that are mostly greyscale images as seen below:

![BWcluster](./coco/images/b_w_cluster.png)

Upon further look, most of the samples are structured with 3 rgb channels as colored images (most of the samples are colored red from 'is_colored' metadata): 
![](https://hackmd.io/_uploads/SyOlVbLbc.png)

However, most of their red green blue components have equal intensity in RGB space.

We see that the loss is lower on colored images that holds more information than the greyscale images:  

![](https://hackmd.io/_uploads/BJQ39eTWc.png)


#### 'Vehicle-like' clusters
Tom's sugggestion:
**Due to the pre-trained feature extractor our model is able to extract multiple semantically meaningful clusters:**
?
The system was able to extract clusters characterized by categories that the net didn't train on as further seen. A reasonable explanation is that the pre-trained encoder we use was trained on these classes and that is why it is extracting those feature classes.  

#### Bicycle cluster 
 ![](https://hackmd.io/_uploads/rkZpLcfxq.png)

#### Bus cluster 
![](https://hackmd.io/_uploads/HJ_N-2Qb9.png)
Tom's suggestion:

**In addition to the high concentration of buses and cars, we also see buildings and poles as part of the same cluster.**
?
Examining the attention map on the features that make this cluster similar, we see features that are not directly related to vehicles (towers, etc.):

![](https://hackmd.io/_uploads/SyFIDsNbc.png)


## Vehicle Supercategory Model
Tom's suggestion:

**Our previous model tries to segment cars as a seperate class from truck & bus (which are labeled background). Often, we need to segment the entire vehicle SuperCategory (SC) together. Here, we train a SC model that tries to segment vehicles, person, and background.**
?
From the vehicle-like clusters we found, we can conclude that giving the model to segment other vehicles (which aren't car) as background might negatively affect the performance. That leads us trying to redefine our categories classes to a more generic definition: `vehicle` with `person`. 

Evaluating the new model on a Super Category (SC) labeled dataset we get:

| Dataset | Mean IoU Person |
| -------- |  -------- |
| Category Model     | 0.309
| Super Category Model |  0.319 |

We slightly improve the performance on person mean IoU.



The top chart is of the `car` category model and the bottom is the `vehicle` super-category model. For the `car` category model the average loss decreases when the number of busses in an image increases, for the train set and the validation set. When there are more bus objects within the image the model is less likely to mistake on the `car` category.
As for the `vehicle` super-category model, the loss on the `vehicle` category increases when the number of bus objects increases.


<span style="color:blue">TODO explain why we improved (hopefully) prediction on buses category? explain the plot?</span>.


#### Cluster Analysis

Fetching similars to one of the vehicles as expected result in a more homogenous cluster (composed of cars + buses).

<span style="color:blue">Todo: add images of cluster</span>

One of the strongest shared features in this cluster are the wheels:
![](https://hackmd.io/_uploads/BJhkzm5xq.png)

Reviewing the attention map reveals a possible confusion: round objects (camera lens) could be categorized as cars due to their similarity to wheels:
![](https://hackmd.io/_uploads/rkGf-X5l5.png)
![](https://hackmd.io/_uploads/SJEV-7cxq.png)
![](https://hackmd.io/_uploads/H1B_Zm9l5.png)
![](https://hackmd.io/_uploads/ryA5WXqxc.png)
![](https://hackmd.io/_uploads/rJF3bQ5eq.png)


#### Additional cluster: Sport cluster

![](https://hackmd.io/_uploads/HyX4YwU-9.png)
![](https://hackmd.io/_uploads/ByXm6Sjb9.png)
![](https://hackmd.io/_uploads/BkUuaHi-q.png)

These clusters show that the model was able to learn the context of the image or at least in case of sport activites the images are charachterized with similar featurs.

#### False and Ambiguous Labels

<span style="color:blue">TODO using cluster + dashboard to highlight COCO/Semantic-seg issues</span>.

With the help of the Sample Analysis feature, we can find some ambiguous labels and false labeling images that we can choose to exclude from our train dataset to improve the performance.

GT: Mislabeled Image: two missing people:
![](https://hackmd.io/_uploads/By630p3b5.png)


- The driver and the woman from the left are segmented as the car.

Prediction: When the model correctly segments all three people.
![](https://hackmd.io/_uploads/Bylo0T3Z9.png)


- Sample error analysis revealed that it wasn't the model's false prediction on that sample but rather false labeling.

Inaccurate GT:
![](https://hackmd.io/_uploads/H1WhQvjbq.png)

Ambiguous label image: toy car as a car :)
![](https://hackmd.io/_uploads/rJuxnD9e5.png)

Challenging image: low light
![](https://hackmd.io/_uploads/B1BT5vqgc.png)

Challenging image: crowds with small people
![](https://hackmd.io/_uploads/ry_IoP5x5.png)


#### Performance and Metadata Analysis

We can plot using Tensorleap the metadata we extract to identify trends and factors that are correlated to the performance. 

![](https://hackmd.io/_uploads/H12-h1rWc.png)

We can see that when the number of person instances per image increases the Cross-Entropy Loss increases and the mean IoU decreases. The net's predictions are less accurate when the image is denser with objects. [rephrase](/3vnw7Z7GSPGBR7R-qITpQQ) 

For the vehicle category, we can see that model performance decreases when the vehicle average size increases. 
![](https://hackmd.io/_uploads/r1R7RkHb5.png)

For all categories: total instances count: [remove](/kMuTbQYySvuV_nOgp0gDSQ)
![](https://hackmd.io/_uploads/rJElUv8b9.png)


Object Average size:


![](https://hackmd.io/_uploads/r1Azdw8Z9.png)
When the average person size per image increase the mean IoU on a person decreases. The segmentation is less accurate.


#### Summary and Conclusion

We have shown how Tensorleap features can reveal mislabeled samples that may confuse the model during training, and metadata factors that affect the performance. All these insights can improve and significantly decrease the model optimization process.

<span style="color:blue">TODO</span>.
