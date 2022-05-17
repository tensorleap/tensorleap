from pathlib import Path
from coco_pytorch.model import UnetSegModel
import tensorflow as tf

OUTPUT_CLASSES = 4

def leap_save_model(target_file_path: Path):
    # Load your model
    # Save it to the path supplied as an argument (has a .h5 suffix)
    model = UnetSegModel()
    # model.compile(optimizer='adam',
    #               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])
    model.save(target_file_path)
