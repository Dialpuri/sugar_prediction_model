import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
import csv
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm.keras import TqdmCallback
import model as networks


@dataclass
class Params:
    dataset_base_dir: str = "./dataset"
    shape: int = 32


def sample_generator(dataset: str = "train"):
    datasets = {"train": "./data/hog/train_hog.csv", "test": "./data/hog/test_hog.csv"}

    classification = {
        "protein": 0,
        "sugar": 1,
        "base": 2,
        "phosphate": 3
    }

    while True: 
        with open(datasets[dataset]) as f:
            r = csv.reader(f)
            for index, row in enumerate(r):
                if index == 0: 
                    continue
                
                class_ = classification[row[-1]]

                class_one_hot = tf.one_hot(class_, len(classification.keys()))

                yield row[:-1], class_one_hot
            
def train():
    num_threads: int = 128
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    tf.config.threading.set_inter_op_parallelism_threads(int(num_threads / 2))
    tf.config.threading.set_intra_op_parallelism_threads(int(num_threads / 2))

    _train_gen = sample_generator("train")
    _test_gen = sample_generator("test")

    input = tf.TensorSpec(shape=(18), dtype=tf.float32)
    output = tf.TensorSpec(shape=(4), dtype=tf.int64)

    train_dataset = tf.data.Dataset.from_generator(
        lambda: _train_gen, output_signature=(input, output)
    )
    test_dataset = tf.data.Dataset.from_generator(
        lambda: _test_gen, output_signature=(input, output)
    )

    model = networks.model()

    optimiser = tf.keras.optimizers.Adam(learning_rate=1e-5)

    model.compile(
        optimizer=optimiser,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    reduce_lr_on_plat = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.8,
        patience=5,
        verbose=1,
        mode="auto",
        cooldown=5,
        min_lr=1e-7,
    )
    epochs: int = 100
    batch_size: int = 8
    steps_per_epoch: int = 1_000_000
    validation_steps: int = 10_000
    name: str = "hog_model_1"

    weight_path: str = f"models/{name}.best.hdf5"

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        weight_path,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
        save_weights_only=False,
    )

    train_dataset = train_dataset.repeat(epochs).batch(batch_size=batch_size)

    test_dataset = test_dataset.repeat(epochs).batch(batch_size=batch_size)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=f"./logs/{name}", histogram_freq=1, profile_batch=(10, 30)
    )

    callbacks_list = [
        checkpoint,
        reduce_lr_on_plat,
        TqdmCallback(verbose=2),
        tensorboard_callback,
    ]

    model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
        verbose=0,
        use_multiprocessing=True,
    )

    model.save(f"models/{name}")


if __name__ == "__main__":
    param = Params()
    
    train()

    # for x in sample_generator("train"):
    #     i, o = x
        
    #     print(i, o)
