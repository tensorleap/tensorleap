import matplotlib.pyplot as plt
from tensorleap import *


def iter_all_samples(data: List[PreprocessResponse]):
    err_i = 0
    for sub in data:
        for i in range(sub.length):
            try:
                x = input_image(i, sub)
                mask = ground_truth_mask(i, sub)
                city_val = metadata_city(i, sub)
                err_i = 0
            except Exception as e:
                print(f"Error in processing sample {i} from {sub['subset_name']} set")
                err_i += 1
                if err_i > 10:
                    return
                continue




def plot_mask(img, mask):
    fig, ax = plt.subplots()
    ax.imshow(img)
    mask = np.argmax(mask, -1) if mask.shape[-1] != 1 or mask.shape[-1] != 3 else mask
    ax.imshow(mask, alpha=0.5)
    plt.show()



if __name__ == "__main__":

    data_subsets: List[PreprocessResponse] = subset_images()  # preprocess and get data preprocess response list
    i, train = 0, data_subsets[0]
    img = input_image(i, train)
    mask = ground_truth_mask(i, train)
    plot_mask(img, mask)     # plot the GT
    iter_all_samples(data_subsets)

    MODEL_PATH = None   # TODO: FILL here your path model
    # Infer
    if MODEL_PATH is not None:
        model = tf.keras.models.load_model(MODEL_PATH)
        x = np.expand_dims(img, axis=0)
        pred = model(x)[0, ...]     # infer
        plot_mask(img, pred)        # plot the model's prediction
        vis_obj = loss_visualizer(img, mask, pred)       # run the custom visualizer
        plt.imshow(vis_obj.data)    # plot the custom heatmap






