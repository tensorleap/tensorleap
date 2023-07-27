

def check_custom_integration():
    responses = preprocess_func_leap()
    path = "/Users/chenrothschild/repo/tensorleap/examples/cifar10/tests"
    os.chdir(path)
    model = os.path.join(path, 'resnet.h5')
    resnet = tf.keras.models.load_model(model)  # load onnx model
    for i in range(0, 20):
        concat = np.expand_dims(input_encoder_leap(i, responses[0]), axis=0)
        y_pred = resnet([concat])
        gt = np.expand_dims(gt_encoder(i, responses[0]), axis=0)
        y_true = tf.convert_to_tensor(gt)
        ls = CategoricalCrossentropy()(y_true, y_pred).numpy()
        sample_index = metadata_sample_index(i, responses[0])
        gt_label_leap = metadata_gt_label_leap(i, responses[0])
        label_name_leap = metadata_label_name_leap(i, responses[0])
        fly_leap = metadata_fly_leap(i, responses[0])
        animal_leap = metadata_animal_leap(i, responses[0])
if __name__ == '__main__':
    check_custom_integration()