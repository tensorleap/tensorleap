
import numpy as np
from scipy.ndimage import zoom

def input_encoder(image: np.ndarray) -> np.ndarray:
    """
    Description: Encodes the input image by resizing it and normalizing its values.
    :param:
    image (np.ndarray): Input image as a numpy array.
    :return:
    resized_image_array (np.ndarray): Encoded and resized image as a numpy array.
    """
    resized_image_array = zoom(image, (7, 7, 1)) / 255
    return resized_image_array