# MNIST


We show an example of building a cnn network trained on MNIST dataset.

This example includes the following steps:

- Loading MNIST dataset and do some very basic preprocessing
- Building a CNN network using tensorflow keras modules
- Sanity Check: testing the (untrained) model prediction inference on local environment.
- Setting up Leap env
- Integrate the model and dataset into the TL.

The model achieved accuracy of 0.98~. 

## Population Exploration

This embedding space is a representation based on the network's extracted features. The samples are from the training set.

<img alt="img.png" height="300" src="images/img_7.png" width="800"/>

We can see that there is a nice separation of the samples based on their classes.
The samples are colored based on their GT class and the dot size is based on the network's error loss. 

Now, we plot the validation set:

<img alt="img.png" height="400" src="images/img_2.png" width="700"/>

We see that there is still a nice separation, however, with more high error loss samples. 
Interesting, those false predictions (larger dot size) are located within clusters of classes which are different than their GT.  


In the digits image space there are some classes which are close to each other. For example, class 1 and class 7 tends to look alike. We can see that from the population exploration plot below, these classes samples are closer 
to each other and there are some false prediction between the samples on the edge. Samples in light blue are from class 1 and the peach colored are from 7.

<img alt="img_10.png" height="500" src="images/img_1.png" width="800"/>


# Error Analysis


<img alt="img.png" height="400" src="images/img_3.png" width="600"/> <img alt="img.png" height="400" src="images/img_4.png" width="600"/>


**Sample 11701:** Prediction: 6, GT: 5

- The loss on that sample is relatively higher than the other samples. From metadata, we get that the sample is closer to 6 class (6.83) than its GT - 5 (7.71). That is why the model predicted the sample to be 6. 




### Average Euclidean Difference Metadata and Average Loss

<img alt="img_16.png" src="images/img_6.png"/>

- We see how when the euclidean difference of a sample from its class centroid, increases, the average loss increases. 

- Euclidean difference from the class centroid is calculated as follows: 
  - We extract the class centroid per class: we take all images in respect to their class and calculate an average per pixel. The output will be an image 28X28.     
  - Per sample, we calculate the euclidean difference from its class centroid. 
  


### Results

![img.png](images/img_5.png)

