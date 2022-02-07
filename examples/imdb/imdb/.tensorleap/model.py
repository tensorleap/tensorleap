from pathlib import Path
from imdb.model_infer import tensorleap_model


def leap_save_model(target_file_path: Path):
    # Load your model
    # Save it to the path supplied as an arugment (has a .h5 suffix)
    model = tensorleap_model()
    model.save(target_file_path)
    print(f'Saving the model as {target_file_path}. Safe to delete this print.')
