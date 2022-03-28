# Adapted from https://www.tensorflow.org/tutorials/images/segmentation

import tensorflow as tf
from model import unet_model
import numpy as np
from skimage import transform
from skimage.io import imread


def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  return pred_mask[0]


def load_image(filename):
   image = imread(filename)
   image = transform.resize(image, (128, 128, 3), order=1, anti_aliasing=True) # Order=1 <=> bilinear
   image = np.expand_dims(image, axis=0)
   return image


def infer_model():
    OUTPUT_CLASSES = 3

    model = unet_model(output_channels=OUTPUT_CLASSES)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    img = load_image("./test_image/COCO_train2014_000000362499.jpg")
    res = model.predict(img)
    print("inferred succesfuly on a single coco image")


if __name__ == "__main__":
    infer_model()
