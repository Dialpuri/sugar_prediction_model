import os
import numpy as np
import pandas as pd
import gemmi
from dataclasses import dataclass
from typing import List
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm.keras import TqdmCallback

import generate_samples as generate_samples
import unet
import enum

@dataclass
class Params:
    dataset_base_dir: str = "./dataset_1.5"
    shape: int = 32

class Types(enum.Enum): 
    sugar: int = 1
    phosphate: int = 2
    base: int = 3


def sample_generator(dataset: str = "train"):
    datasets = {"train": "./data/1.5A_radius/train_dataset_1.5_calpha_2.csv", "test": "./data/1.5A_radius/test_dataset_1.5_calpha_2.csv"}

    df: pd.DataFrame = pd.read_csv(datasets[dataset])
    df: pd.DataFrame = df.astype({'X': 'int', 'Y': 'int', 'Z': 'int'})
    df_np: np.ndarray = df.to_numpy()

    def get_density(path: str, translation: List[int]) -> np.ndarray:

        assert len(translation) == 3

        map: gemmi.FloatGrid = gemmi.read_ccp4_map(path).grid
        array: np.ndarray = np.array(
            map.get_subarray(
                start=translation, shape=[param.shape, param.shape, param.shape]
            )
        ) 
        array = array.reshape(param.shape, param.shape, param.shape, 1)
        return array

    def get_atom_density(path: str, translation: List[int]) -> np.ndarray: 
    
        map: gemmi.FloatGrid = gemmi.read_ccp4_map(path).grid
        array: np.ndarray = np.array(
            map.get_subarray(
                start=translation, shape=[param.shape, param.shape, param.shape]
            )
        ) 
        hot_array = tf.one_hot(array, depth=2)
        return hot_array

    while True:
        for candidate in df_np:
            assert len(candidate) == 4

            pdb_code: str = candidate[0]
            translation: str = candidate[1:4]
            
            density_path: str = os.path.join(
                param.dataset_base_dir, pdb_code, f"{pdb_code}{names.density_file}.map"
            )
            raw_density = get_density(density_path, translation)

            if atom_type == Types.base:
                base_path: str = os.path.join(
                    param.dataset_base_dir, pdb_code, f"{pdb_code}{names.base_file}.map"
                )
                base_array = get_atom_density(base_path, translation)
                yield raw_density, base_array

            if atom_type == Types.sugar:
                sugar_path: str = os.path.join(
                    param.dataset_base_dir, pdb_code, f"{pdb_code}{names.sugar_file}.map"
                )
                sugar_array = get_atom_density(sugar_path, translation)
                yield raw_density, sugar_array

            if atom_type == Types.phosphate:
                phosphate_path: str = os.path.join(
                    param.dataset_base_dir, pdb_code, f"{pdb_code}{names.phosphate_file}.map"
                )
                phosphate_array = get_atom_density(phosphate_path, translation)
                yield raw_density, phosphate_array
            
            
def train():
    num_threads: int = 128
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

    es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6)

    epochs: int = 25
    batch_size: int = 8
    steps_per_epoch: int = 10000
    validation_steps: int = 1000
    name: str = "sugar_model_1"

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
        es
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

    atom_type = Types.sugar
    train()
