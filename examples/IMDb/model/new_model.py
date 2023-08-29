import tensorflow as tf

from IMDb.config import CONFIG

if __name__=='__main__':
    vectorized_inputs = tf.keras.Input(shape=250, dtype="int64")
    x = tf.keras.layers.Embedding(CONFIG['MAX_FEATURES'] + 1, CONFIG['SEQUENCE_LENGTH'])(vectorized_inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(28, activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    output = tf.keras.layers.Dense(2, activation='softmax')(x)

    model = tf.keras.Model(inputs=vectorized_inputs, outputs=output)
    model.save('imdb-dense.h5')