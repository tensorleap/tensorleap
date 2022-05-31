# Semantic Segmentation - Pytorch Implementation

In this example, we demonstrate the use of Tensorleap on a Computer Vision task - Semantic Segmentation with COCO dataset (**coco 14** training and validation data files.)

## The Task

The goal of semantic segmentation is to label each pixel of an image with a corresponding class. This task is commonly referred to as dense prediction because the prediction is for every pixel in the image. In semantic segmentation, the boundary of objects is labeled with a mask, and object classes are labeled with a class label. The output is a high-resolution image (typically of the same size as the input image) in which each pixel is classified to a particular class.

## The Dataset

[**COCO**](https://cocodataset.org) (Common Objects in Context) is a large-scale object detection, segmentation, and captioning dataset with 80 classes.

## The Models

The models are based Pytorch implementation:

1. Deeplabv3-ResNet 

Pre-trained model is constructed by a Deeplabv3 model using a ResNet-50

[Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)

2. A simplified representation of [U-Net network](https://arxiv.org/abs/1505.04597v1)



