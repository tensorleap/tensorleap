import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.models import Model
import numpy as np


# define baseline cnn model for mnist
def build_model() -> tf.keras.Model:
    input_shape = (28, 28, 1)
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Convert to functional Model API
    inputs = tf.keras.layers.Input(shape=input_shape)
    model = tf.keras.models.Model(inputs, model.call(inputs))

    return model


# run model one one randomly selected sample
def model_infer_one_sample(data: np.ndarray, model: tf.keras.Model):
    idx = np.random.choice(len(data))
    x = np.expand_dims(data[idx], axis=0)
    return model(x)

