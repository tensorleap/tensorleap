from pathlib import Path
from imdb.model_infer import tensorleap_conv_model, tensorleap_dense_model, tensorleap_masked_conv_model
MODEL_TYPES = ["dense", "masked dense", "convolution"]


def leap_save_model(target_file_path: Path):
    # Load your model
    # Save it to the path supplied as an arugment (has a .h5 suffix)
    chosen_model = MODEL_TYPES[1]

    if chosen_model == MODEL_TYPES[0]:
        model = tensorleap_dense_model()
    elif chosen_model == MODEL_TYPES[1]:
        model = tensorleap_masked_conv_model()
    elif chosen_model == MODEL_TYPES[2]:
        model = tensorleap_conv_model()
    model.save(target_file_path)
    print(f'Saving the model as {target_file_path}. Safe to delete this print.')
