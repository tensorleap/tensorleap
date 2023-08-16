from typing import Tuple, List
import webcolors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from code_loader.contract.responsedataclasses import BoundingBox

from cityscapes_od.data.preprocess import Cityscapes
from cityscapes_od.utils.general_utils import normelized_polygon


def rgb_to_color_name(rgb_value: Tuple[int]) ->str:
    """
    Description: This function takes an RGB color value as input and converts it into a color name.
    Input: rgb_value (Tuple[int]): An RGB color value represented as a tuple of three integers (red, green, blue).
    Output: color_name (str): The color name corresponding to the input RGB value or 'r' as a default placeholder.
    """
    try:
        color_name = webcolors.rgb_to_name(rgb_value)
    except ValueError:
        color_name = 'r'
    return color_name

def plot_image_with_bboxes(image: np.ndarray, bounding_boxes: List[BoundingBox], type: str):
    """
    Description: The function takes an image and a list of bounding boxes as input and visualizes the image with
    bounding boxes overlaid.
    Input: image (numpy array): Input RGB image as a NumPy array.
           bounding_boxes (list): List of bounding boxes represented as [x_center, y_center, width, height, label].
           type (str): The bboxes type- gt ot prediction.
    Output: None. The function directly displays the image with overlaid bounding boxes using Matplotlib.
    """
    # Create a figure and axis
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image)

    # Add bounding boxes to the plot
    for bbox in bounding_boxes:
        label = bbox.label
        class_id = Cityscapes.get_class_id(label)
        color = Cityscapes.get_class_color(class_id)
        color_name = rgb_to_color_name(color)

        if label != 'unlabeled':
            x_center, y_center, width, height = bbox.x, bbox.y, bbox.width, bbox.height
            x_min = x_center - width / 2
            y_max = y_center + height / 2
            # Convert relative coordinates to absolute coordinates
            x_abs = x_min * image.shape[1]
            y_abs = y_max * image.shape[0]
            width_abs = width * image.shape[1]
            height_abs = -(height * image.shape[0])

            # Create a rectangle patch and add it to the plot
            rect = patches.Rectangle((x_abs, y_abs), width_abs, height_abs, linewidth=1, edgecolor=color_name, facecolor='none')
            ax.add_patch(rect)

            # Add label text to the rectangle
            plt.text(x_abs, y_abs, label, color=color_name, fontsize=8, backgroundcolor='white')

    # Show the plot
    plt.title(f"Image with {type} bboxes")
    plt.show()

def plot_image_with_polygons(image_height: int, image_width: int, polygons, image: np.ndarray):
    """
    Description: The function takes an image and a list of polygons as input and visualizes the image with
    polygons overlaid.
    Input: image (numpy array): Input RGB image as a NumPy array.
           image_height :(int): Height of the input image in pixels.
           image_width: (int): Width of the input image in pixels.
           polygons (list): List of polygons represented as dictionaries, with 'label' (int) and 'polygon' (list) keys.
    Output: None. The function directly displays the image with overlaid polygons using Matplotlib.
    """
    # Create a figure and axis
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image)

    # Add polygons to the plot
    for polygon in polygons:
        polygon = normelized_polygon(image_height, image_width, polygon)
        label = polygon['label']
        color = Cityscapes.get_class_color(label)
        color_name = rgb_to_color_name(color)

        class_name = Cityscapes.get_class_name(label)
        coords = polygon['polygon']

        # Create a polygon patch and add it to the plot
        poly_patch = patches.Polygon(coords, linewidth=1, edgecolor=color_name, facecolor='none')
        ax.add_patch(poly_patch)

        # Add label text to the polygon
        centroid = [sum(coord[0] for coord in coords) / len(coords), sum(coord[1] for coord in coords) / len(coords)]
        plt.text(centroid[0], centroid[1], class_name, color=color_name, fontsize=8, backgroundcolor='white')

    plt.title("Image with Polygons")
    plt.show()