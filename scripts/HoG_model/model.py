import tensorflow as tf
import tensorflow_addons as tfa


def model(): 
    inputs = x = tf.keras.Input(shape=(18,))
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(48, activation='relu')(x)
    x = tf.keras.layers.Dense(4, activation='softmax')(x)
    return tf.keras.Model(inputs, x)