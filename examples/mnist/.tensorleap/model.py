from pathlib import Path
from mnist.model import build_model


def leap_save_model(target_file_path: Path):
    # Load your model
    model = build_model()
    # Save it to the path supplied as an argument (has a .h5 suffix)
    model.save(target_file_path)
    print(f'Saving the model as {target_file_path}. Safe to delete this print.')
