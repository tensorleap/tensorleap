from pathlib import Path
from imdb.model_infer import tensorleap_model_with_attention


def leap_save_model(target_file_path: Path):
    # Load your model
    # Save it to the path supplied as an arugment (has a .h5 suffix)
    model = tensorleap_model_with_attention()
    model.save(target_file_path)
    print(f'Saving the model as {target_file_path}. Safe to delete this print.')
