# Image Analysis

## The MNIST Dataset

In this example, we used the [**MNIST dataset**](http://yann.lecun.com/exdb/mnist/) (handwritten digits) to train a convolutional neural network (CNN) for image classification, and reached a validation accuracy of \~0.98 on our trained model.

### Population Exploration <a href="#population-exploration" id="population-exploration"></a>

Below is a **population exploration** plot. It represents a samples' similarity map based on the model's latent space, built using the extracted features of the trained model.

#### Training Set:

The samples shown below were taken from the **training set** and are colored based on their **ground truth** class. The dot's size represents the model's error loss for that specific sample. We can see that there is a clear separation of the samples based on their classes.

![Population Exploration Analysis of Training Samples](<../.gitbook/assets/image (32).png>)

#### Validation Set:

The plot below is a similarity map within the **validation set** and we can see that there is still a nice separation, but there are quite a few samples with higher error loss (larger dot size). Interestingly, those false predictions are located within clusters of classes different from their **ground truth** class.

![Population Exploration of the Validation Set](<../.gitbook/assets/image (19) (1).png>)

#### Similar Classes

In the visualization below we can see that the samples from class **7** _(peach-colored)_ and class **1** _(light-blue)_ are relatively close to each other. This makes sense since the digit **7** is visually similar to the digit **1**.

![Close Similar Clusters of Classes 1 and 7](../.gitbook/assets/mnist-similar-classes-2.gif)

### Error Analysis <a href="#error-analysis" id="error-analysis"></a>

The dot's size in the visualization below represents the sample's loss. The selected sample is a challenging one, and we can also see that it has a relatively high loss. The sample's metadata shown on the right panel shows a distance of 6.83 to class `6` and 7.71 to the ground truth - class `5,`and indeed prediction was class `6`.

![Prediction: 6](../.gitbook/assets/img_3.png) ![GT:5](../.gitbook/assets/img_4.png)

### Sample Analysis <a href="#sample-analysis" id="sample-analysis"></a>

The Tensorleap platform provides a way to further explore the model's response to specific data samples. For example, from the results below, we can see that the model's prediction was **9** while the ground truth is **8**. Additionally, we can use the heat map to see which features of the data contributed to each of the output classes.

![Heat Maps Correlated for each Output](../.gitbook/assets/mnist-sample-analysis.gif)

### Fetch Similars <a href="#fetch-similars" id="fetch-similars"></a>

As explained, Tensorleap tracks how each learned feature responds to each sample. This allows finding samples that the model considers similar, as they _activate_ similar learned features.

This functionality also allows to automatically find **candidates for labeling**, detect **ambiguities** in the data and **mislabeled** samples.&#x20;

#### Fetch Similars to a Sample - Example

In the example below we ran a **Fetch Similars** operation to find similar samples to the sample <img src="../.gitbook/assets/image (26) (1).png" alt="" data-size="line">. The result is a cluster of similar samples:

![Fetch Similar Results](<../.gitbook/assets/image (16) (2).png>) ![Similar Samples](<../.gitbook/assets/image (13) (1) (1) (1) (1) (1) (1).png>)

#### Example - Investigating the "Closed" 4 Digit

The MNIST dataset includes samples of the digit 4 written in two ways: <img src="../.gitbook/assets/image (7) (1) (1).png" alt="" data-size="line"> and <img src="../.gitbook/assets/image (17) (1) (1) (1) (1) (1) (1).png" alt="" data-size="line">. \
Running the Fetch Similars operation to find samples that are similar to <img src="../.gitbook/assets/image (7) (1) (1).png" alt="" data-size="line"> returns similar samples with presented heat maps that highlight the features that define these similarities.

![Similar Samples Heat Map](<../.gitbook/assets/image (20) (1) (1) (1) (1) (1).png>)

In the 4 digit class the majority of the samples are written in the open form <img src="../.gitbook/assets/image (17) (1) (1) (1) (1) (1) (1).png" alt="" data-size="line">. The operation detected four samples that are similar to the closed form <img src="../.gitbook/assets/image (7) (1) (1).png" alt="" data-size="line">, while the other samples are mostly the digit `9`. This provides us with useful insights about the model and data and how to improve them. For example, we can:

- Separate the 4 class into two classes - open and closed digits.
- Enrich the data with additional samples of closed digits.
- Remove the closed digits from the dataset and support only the open form.

Similar analysis can be done for the two ways of writing the digit seven: <img src="../.gitbook/assets/image (27) (1) (1) (1) (1) (1).png" alt="" data-size="line"> and <img src="../.gitbook/assets/image (23) (2) (1).png" alt="" data-size="line">.&#x20;

### Metrics Analysis <a href="#results-analysis" id="results-analysis"></a>

The Tensorleap Dashboard allows users to visualize the training progress and performance metrics of their models. Additionally, users can explore correlations and trends between various metadata to gain insights into their data.

**Average Euclidean Distance Metadata vs Average Loss**

For each class, its **mean image** is calculated (28x28) by averaging each pixel. For each sample, its Euclidean distance from the **mean image** of its class is calculated and set as its metadata.

The Dashboard allows us to visualize and investigate each metadata interactively. The visualization below shows the average error loss per class, vs the Euclidean distance.

![Average Euclidean Distance vs Average Loss (click-to-zoom)](<../.gitbook/assets/image (25).png>)

This plot reveals a strong correlation between the Euclidean distance and the average loss - as the Euclidean distance increases, the loss increases with it. This is expected, as the model is less capable of generalizing samples with high irregularities.

**Loss and Accuracy Visualizations**

During training, metrics are collected and presented in the Dashboard.\
For example, a visualization of the model's Loss (left) and Accuracy(right) vs batches:

![Loss/Accuracy vs Batch](<../.gitbook/assets/image (27) (1).png>)

As we can see, the model convergences well and achieves an accuracy of \~0.98 on the **validation** set.

## Summary

The **Tensorleap** platform provides powerful tools for analyzing and understanding deep learning models. In this example, we presented only a few examples of the types of insights that can be gained using the platform.&#x20;
