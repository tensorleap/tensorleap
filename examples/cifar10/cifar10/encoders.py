
import numpy as np
from scipy.ndimage import zoom

def input_encoder(image: np.ndarray) -> np.ndarray:
    resized_image_array = zoom(image, (7, 7, 1)) / 255
    return resized_image_array