import os
import numpy as np
import pandas as pd
import gemmi
from dataclasses import dataclass

import tensorflow as tf
import tensorflow_addons as tfa
from tqdm.keras import TqdmCallback

import generate_samples as generate_samples
import unet


@dataclass
class Params:
    dataset_base_dir: str = "./dataset_0.75"
    shape: int = 32


def sample_generator(dataset: str = "train"):
    datasets = {"train": "./data/0.75A_rad/test.csv", "test": "./data/0.75A_rad/train.csv"}

    df: pd.DataFrame = pd.read_csv(datasets[dataset])
    df: pd.DataFrame = df.astype({'X': 'int', 'Y': 'int', 'Z': 'int'})
    df_np: np.ndarray = df.to_numpy()

    while True:
        for candidate in df_np:
            assert len(candidate) == 4

            pdb_code: str = candidate[0]
            X: int = candidate[1]
            Y: int = candidate[2]
            Z: int = candidate[3]

            density_path: str = os.path.join(
                param.dataset_base_dir, pdb_code, f"{pdb_code}{names.density_file}.map"
            )
            sugar_path: str = os.path.join(
                param.dataset_base_dir, pdb_code, f"{pdb_code}{names.sugar_file}.map"
            )
            phosphate_path: str = os.path.join(
                param.dataset_base_dir,
                pdb_code,
                f"{pdb_code}{names.phosphate_file}.map",
            )
            base_path: str = os.path.join(
                param.dataset_base_dir, pdb_code, f"{pdb_code}{names.base_file}.map"
            )
            no_sugar_path: str = os.path.join(
                param.dataset_base_dir, pdb_code, f"{pdb_code}{names.no_sugar_file}.map"
            )

            density_map: gemmi.FloatGrid = gemmi.read_ccp4_map(density_path).grid
            sugar_map: gemmi.FloatGrid = gemmi.read_ccp4_map(sugar_path).grid
            phosphate_map: gemmi.FloatGrid = gemmi.read_ccp4_map(phosphate_path).grid
            base_map: gemmi.FloatGrid = gemmi.read_ccp4_map(base_path).grid
            no_sugar_map: gemmi.FloatGrid = gemmi.read_ccp4_map(no_sugar_path).grid

            density_array: np.ndarray = np.array(
                density_map.get_subarray(
                    start=[X, Y, Z], shape=[param.shape, param.shape, param.shape]
                )
            )
            # sugar_array: np.ndarray = np.array(
            #     sugar_map.get_subarray(
            #         start=[X, Y, Z], shape=[param.shape, param.shape, param.shape]
            #     )
            # )
            phosphate_array: np.ndarray = np.array(
                phosphate_map.get_subarray(
                    start=[X, Y, Z], shape=[param.shape, param.shape, param.shape]
                )
            )
            # base_array: np.ndarray = np.array(
            #     base_map.get_subarray(
            #         start=[X, Y, Z], shape=[param.shape, param.shape, param.shape]
            #     )
            # )
            # no_sugar_array: np.ndarray = np.array(
            #     no_sugar_map.get_subarray(
            #         start=[X, Y, Z], shape=[param.shape, param.shape, param.shape]
            #     )
            # )

            density_yield: np.ndarray = density_array.reshape(
                param.shape, param.shape, param.shape, 1
            )

            # sugar_array_hot = tf.one_hot(sugar_array, depth=2)
            phosphate_array_hot = tf.one_hot(phosphate_array, depth=2)
            # print(sugar_array_hot)

            # output_yield = np.stack(
            #     (no_sugar_array, sugar_array, phosphate_array, base_array), axis=-1
            # )

            yield density_yield, phosphate_array_hot


def train():
    num_threads: int = 128
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    tf.config.threading.set_inter_op_parallelism_threads(int(num_threads / 2))
    tf.config.threading.set_intra_op_parallelism_threads(int(num_threads / 2))

    _train_gen = sample_generator("train")
    _test_gen = sample_generator("test")

    input = tf.TensorSpec(shape=(32, 32, 32, 1), dtype=tf.float32)
    output = tf.TensorSpec(shape=(32, 32, 32, 2), dtype=tf.int64)

    train_dataset = tf.data.Dataset.from_generator(
        lambda: _train_gen, output_signature=(input, output)
    )
    test_dataset = tf.data.Dataset.from_generator(
        lambda: _test_gen, output_signature=(input, output)
    )

    model = unet.binary_model()

    loss = tfa.losses.sigmoid_focal_crossentropy

    optimiser = tf.keras.optimizers.Adam(learning_rate=1e-5)

    model.compile(
        optimizer=optimiser,
        loss=loss,
        metrics=["accuracy", "categorical_accuracy"],
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
    epochs: int = 25
    batch_size: int = 8
    steps_per_epoch: int = 10000
    validation_steps: int = 1000
    name: str = "phos_0.75"

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
    names = generate_samples.Names()
    param = Params()

    train()
