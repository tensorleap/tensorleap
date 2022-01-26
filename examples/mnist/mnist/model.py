import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Flatten, Dense, BatchNormalization
import numpy as np


# define baseline cnn model for mnist
def build_model() -> tf.keras.Sequential:
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# run model one one randomly selected sample
def model_infer_one_sample(data: np.ndarray, model: tf.keras.Sequential):
    idx = np.random.choice(len(data))
    x = np.expand_dims(data[idx], axis=0)
    return model(x)
