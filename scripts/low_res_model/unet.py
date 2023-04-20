"U-Net - Original Implementation Paul Bond, Edited by Jordan Dialpuri"

import tensorflow as tf
import tensorflow_addons as tfa
from functools import partial

_ARGS = {"padding": "same", "activation": "relu", "kernel_initializer": "he_normal"}
_downsampling_args = {
    "padding": "same",
    "use_bias": False,
    "kernel_size": 3,
    "strides": 1,
}

def binary_model():
    inputs = x = tf.keras.Input(shape=(32, 32, 32, 1))
    skip_list = []

    filter_list = [64, 128, 256, 512, 1024]

    for filters in filter_list:
        x = tf.keras.layers.Conv3D(filters, **_downsampling_args)(x)
        x = tfa.layers.InstanceNormalization(axis=-1,
                                             center=True,
                                             scale=True,
                                             beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv3D(filters, **_downsampling_args)(x)
        x = tfa.layers.InstanceNormalization(axis=-1,
                                             center=True,
                                             scale=True,
                                             beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(x)
        x = tf.keras.layers.ReLU()(x)
        skip_list.append(x)
        x = tf.keras.layers.MaxPool3D(2)(x)

    x = tf.keras.layers.Conv3D(1024, 3, **_ARGS)(x)
    x = tf.keras.layers.Conv3D(1024, 3, **_ARGS)(x)

    for filters in reversed(filter_list):
        x = tf.keras.layers.Conv3DTranspose(filters, 3, 2, padding="same")(x)
        x = tf.keras.layers.concatenate([x, skip_list.pop()])
        x = tf.keras.layers.Conv3D(filters, 3, **_ARGS)(x)
        x = tfa.layers.InstanceNormalization(axis=-1,
                                             center=True,
                                             scale=True,
                                             beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv3D(filters, 3, **_ARGS)(x)
        x = tfa.layers.InstanceNormalization(axis=-1,
                                             center=True,
                                             scale=True,
                                             beta_initializer="random_uniform",
                                             gamma_initializer="random_uniform")(x)
        x = tf.keras.layers.ReLU()(x)

    outputs = tf.keras.layers.Conv3D(2, 3, padding="same", activation="softmax")(x)

    return tf.keras.Model(inputs, outputs)

if __name__ == "__main__":
    model().summary()
