# Semantic Segmentation

In this example, we show the use of Tensorleap on a Computer Vision task - Semantic Segmentation on COCO data. We use `coco 14` training and validation data files combined with a `MobileNetV2` backbone and a `pix2pix` based decoder.

## The Task

The goal of semantic segmentation is to label each pixel of an image with a corresponding class. This task is commonly referred to as dense prediction because the prediction is for every pixel in the image. In semantic segmentation, the boundary of objects is labeled with a mask, and object classes are labeled with a class label. The output is a high-resolution image (typically of the same size as the input image) in which each pixel is classified to a particular class.

## The Dataset

[**COCO**](https://cocodataset.org) (Common Objects in Context) is a large-scale object detection, segmentation, and captioning dataset with 80 classes.

## The Model

The U-Net was developed by Olaf Ronneberger et al. for Biomedical Image Segmentation. The architecture consists of an encoder and a decoder. The encoder captures the image context and the decoder enables precise localization using transposed convolutions. We use a modified U-Net, which is an end-to-end Fully Convolutional Network Model, for the task.

As in the original paper, a simplified U-Net is described as follows:

![U-Net Architecture](<../.gitbook/assets/image (41).png>)

**Transfer Learning**

A pre-trained MobileNetV2 model for the encoder was used. This helps to learn robust features and reduces the number of trainable parameters. The decoder/up-sampler is simply a series of upsample blocks. Only the decoder is trained during the training process, while the encoder weights are frozen.

### Category Prediction Model

The model's task is to segment images consisting of two categories: `person` and `car`.

After evaluating our model on a dataset subset containing cars and persons, we get these performance metrics:

- Mean IoU **Person** **`0.309`**
- Mean IoU **Car** **`0.262`**

### **Population Exploration**

The plot below is a **population exploration** plot. It represents a samples' similarity map based on the model's latent space, built using the extracted features from the neural network.

To qualitatively analyze the model's predictions of the different classes, we utilize Tensorleap's **Population Exploration** analysis.

![Population Exploration](<../.gitbook/assets/image (42).png>)

### **Cluster Analysis**

Selecting samples from different areas of the model's latent space and use the **Fetch Similars** tool to create unique clusters of similar samples, such as:

#### **B\&W cluster**

A cluster with grayed images had been detected. Using the **Fetch Similars** tool, we get these images:

![Grayed Images Cluster](<../.gitbook/assets/image (21).png>)

As seen from the images, this cluster not only captures **grayscale** images but also **RGB** images with a small variation in [**hue**](https://en.wikipedia.org/wiki/Hue).

Viewing the cluster in Tensorleap's cluster analysis tools shows that the vast majority of samples are **RGB** images and not **grayscale** (plotted as red dots below). Additionally, comparing the model's performance on **grayscale** vs **RGB** images yields that, on average, RGB images have lower error loss.

![RGB vs Grayscale Loss Comparison](<../.gitbook/assets/image (23).png>) ![Grayed Cluster - RGB (red) and Grayscale (Blue)](<../.gitbook/assets/image (26).png>)

#### **'Vehicle-Like' Clusters**

Our model's latent spaces have multiple semantically meaningful vehicle clusters, for example, the **Bicycle** and **Bus** clusters:

![Bicycle Cluster (click-to-zoom)](<../.gitbook/assets/image (22).png>) ![Bus Cluster (click-to-zoom)](<../.gitbook/assets/image (21) (1).png>)

Surprisingly, the attention map that highlights cluster defining features contains not only bus features, bus also buildings and towers.

![Bus Cluster Heat-map](<../.gitbook/assets/image (29).png>)

### Vehicle Super-Category Model

When running analysis on Tensorleap, we noticed issues with this model trying to segment **cars** as a separate class from **trucks** and **buses** (which are labeled as **background**).

One possible solution is to segment the entire **Vehicle SuperCategory** together, and this is what is done in this section.

#### **Model Performance**

After training the Vehicle SuperCategory, we get these metrics:

- Mean IoU **Person** **`0.319`**
- Mean IoU **Vehicle** **`0.312`**

#### **Cluster Analysis**

**Fetching Similars** to one of the vehicles, as expected, results in a more homogeneous cluster composed of cars + buses.

![Vehicles Cluster](<../.gitbook/assets/image (35).png>)

Reviewing the attention map (below) shows that the model is able to find strong, discriminative features of this cluster, such as wheels. In addition, it reveals possible confusion as some round objects could be categorized as vehicles, due to their similarities to wheels.

![Vehicles Cluster Heat-map](<../.gitbook/assets/image (9).png>)

The figure below exemplify this confusion. It shows a person inside a car holding a camera. While the **ground truth** is that of a person (bottom right), the actual prediction is that of a car (top right). This is due to the camera lens that provided features that supported a vehicle class (top left) and associated the sample with the cluster (bottom left).

![Sample Analysis](<../.gitbook/assets/image (28).png>)

**Effect on the Person Class**

Using Tensorleap's **Population Exploration** analysis, we can compare the embedding of images with a high percent of car pixels to a high percent of people pixels in the original model (top figures) and the new model (bottom figures):

![Population Exploration Analysis](<../.gitbook/assets/image (19).png>)

Since our new model is now able to use a wider collection of features to describe the **vehicle** category, its latent space provides better separability between **humans** and **vehicles**, and is able to more accurately capture these two categories.

Thus, for example, when we examine the Mean IoU on the person class, we see that our Super Category model is more accurate than the original one:

| Dataset              | Mean IoU Person |
| -------------------- | --------------- |
| Category Model       | 0.309           |
| Super Category Model | 0.319           |

### **Sports Cluster**

Our model is also able to use context to group images, as shown by this cluster containing sports activities**:**

---

![Sports Cluster](<../.gitbook/assets/image (15).png>)

## **Sample Analysis**

### **False and Ambiguous Labels**

With the help of the **Sample Analysis** tool, ambiguous labels and mislabeled images can be detected. These can later be considered for exclusion in order to improve performance.

#### Mislabeling Example

![Mislabeled Ground Truth](<../.gitbook/assets/image (5).png>) ![Model's Prediction](<../.gitbook/assets/image (24).png>)

On the left is the mislabeled **ground truth**, which segmented the driver and one of the women as **car**. On the right is the model's **prediction,** which correctly segmented all three people.

#### Inaccurate Labeling

![Inaccurate Labeling Example](<../.gitbook/assets/image (40).png>)

#### Ambiguous Labeling

In the example below, the toy car is marked as **car**.

![Ambiguous Labeling](<../.gitbook/assets/image (27).png>)

#### &#x20;Challenging Images

In addition, a few challenging images were detected. For example, poor lighting (left) and a crowd of people with very density:

![Low Light](<../.gitbook/assets/image (20).png>) ![Crowd](<../.gitbook/assets/image (12).png>)

## Performance and Metadata Analysis

The Tensorleap Dashboard enables you to see how your data is distributed across various features. This enables us to identify trends and factors that are correlated to the performance.

Below is a visualization of the number of **Persons** (left) and **Vehicles** (right) per image vs **Cross-entropy Loss & Mean IoU:**

![Persons (left) and Vehicles (right) per image vs Cross-entropy Loss & Mean IoU](<../.gitbook/assets/image (2).png>)

From this visualization, it is clear that as the number of **persons** in an image increases, the **Cross-Entropy loss** increases and the **Mean IoU** decreases. Same goes for the **vehicles** category. The model's predictions are less accurate when the image is denser with objects.

## **Summary**

The **Tensorleap** platform provides powerful tools for analyzing and understanding deep learning models. In this example, we presented only a few examples of the types of insights that can be gained using the platform.&#x20;
