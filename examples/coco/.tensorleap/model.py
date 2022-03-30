from pathlib import Path
from coco.model import unet_model
import tensorflow as tf

OUTPUT_CLASSES = 4

def leap_save_model(target_file_path: Path):
    # Load your model
    # Save it to the path supplied as an argument (has a .h5 suffix)
    model = unet_model(output_channels=OUTPUT_CLASSES)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.save(target_file_path)

