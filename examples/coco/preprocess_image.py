from typing import Union
import numpy as np
from cv2 import cv2
import tensorflow as tf

ImageTensorType = tf.Tensor
MaskTensorType = tf.Tensor
KerasModel = tf.keras.Model
MaskType = np.ndarray


def preprocess_image_or_mask(image: Union[ImageTensorType, ImageTensorType],
                             height: int = 256,
                             width: int = 256,
                             scale: float = 255.0,
                             clip_values: bool = False) -> Union[ImageTensorType, ImageTensorType]:
    resized_image = tf.image.resize(
        image, size=(height, width), method="bilinear", preserve_aspect_ratio=True, antialias=False
    )
    padded_image = tf.image.resize_with_pad(resized_image, height, width, antialias=False)
    num_channels = image.shape[2]
    reshaped_image = tf.reshape(padded_image, (height, width, num_channels))
    if clip_values:
        reshaped_image = tf.clip_by_value(reshaped_image, clip_value_max=1, clip_value_min=0)
    else:
        reshaped_image = tf.cast(reshaped_image, dtype=tf.float32)  # pylint: disable= unexpected-keyword-arg
        # Normalize pixels in the input image
        reshaped_image = reshaped_image / scale
    return reshaped_image


def predict_segmentation(model: KerasModel, image: np.ndarray, scale: float = 255.0) -> MaskType:
    reshaped_image = preprocess_image_or_mask(image=image, height=256, width=256,scale=scale)
    image = tf.expand_dims(reshaped_image, axis=0)
    pr_mask = np.uint8(np.round(model.predict(image)) * scale)
    mask = pr_mask.squeeze()
    return mask


def load_model(file_name: str = "solaredge_rooftop_segmentation.h5"):
    # use the model only for inference
    model = tf.keras.models.load_model(file_name, compile=False)
    return model


def calc_iou_score(arr1: MaskType, arr2: MaskType) -> float:
    component1 = arr1.astype(bool)
    component2 = arr2.astype(bool)
    overlap = component1 * component2  # Logical AND
    union = component1 + component2  # Logical OR
    IOU = overlap.sum() / float(union.sum())
    return IOU  # type: ignore


def test_prediction():
    scale: float = 255.0
    image_path = r"test/images/2492689572392.jpg"
    mask_path = r"test/masks/2492689572392.png"
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    mask_image = np.expand_dims(cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY), axis=-1)
    reshaped_mask_image_tf = preprocess_image_or_mask(image=mask_image, height=256, width=256,
                                                      scale=scale, clip_values=True)
    reshaped_mask_image = (reshaped_mask_image_tf.numpy().squeeze() * scale).astype(np.uint8)
    model = load_model()
    pr_mask = predict_segmentation(model=model, image=img)
    iou_score = calc_iou_score(arr1=reshaped_mask_image, arr2=pr_mask)
    print(f"iou_score: {iou_score}")


if __name__ == "__main__":
    test_prediction()
