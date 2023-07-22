
import numpy as np
from scipy.ndimage import zoom

# Input encoder fetches the image with the index `idx` from the `images` array set in
# the PreprocessResponse data. Returns a numpy array containing the sample's image.
def input_encoder(image: np.ndarray) -> np.ndarray:
    resized_image_array = zoom(image, (7, 7, 1)) / 255
    return resized_image_array