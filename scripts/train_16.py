import sys
import time

import import_data as data
from dataclasses import dataclass
import os
import gemmi
import numpy as np
import parameters
import tensorflow as tf
import unet
from keras.metrics import categorical_accuracy, binary_accuracy
import tkinter
import matplotlib
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import pandas as pd
from sklearn.utils import class_weight
import tensorflow_addons as tfa
import tensorflow.keras.backend as K

matplotlib.use('Agg', force=True)
from tqdm.keras import TqdmCallback


def get_map_list(filter_: str) -> list[str]:
    return [path.path for path in os.scandir(params.maps_dir) if filter_ in path.name]



def weighted_cross_entropy_fn(y_true, y_pred):
    
    weights = { 
        0: 0.25,
        1: 0.75
    }

    n_y_pred = y_pred.numpy()
    print(n_y_pred.shape)
    pred = np.argmax(n_y_pred, axis=-1, keepdims=True)
    print(np.unique(pred, return_counts=True))

    tf_y_true = tf.cast(y_true, dtype=y_pred.dtype)
    tf_y_pred = tf.cast(y_pred, dtype=y_pred.dtype)

    weights_v = tf.where(tf.equal(tf_y_true, 1), weights[1], weights[0])
    weights_v = tf.cast(weights_v, dtype=y_pred.dtype)
    ce = K.binary_crossentropy(tf_y_true, tf_y_pred)
    loss = K.mean(tf.multiply(ce, weights_v))
    return loss


def weighted_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0.0, axis=-1):
    weights = {
        # 0: 0.34070462,
        0: 0.001,
        1: 2.15107971,
        2: 10.56569157,
        3: 2.10228326
    }

    weights = np.array(list(weights.values()))

    if isinstance(axis, bool):
        raise ValueError(
            "`axis` must be of type `int`. "
            f"Received: axis={axis} of type {type(axis)}"
        )
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    weighted_predictions = weights * y_pred

    n_y_pred = y_pred.numpy()
    pred = np.argmax(n_y_pred, axis=-1, keepdims=True)
    print(np.unique(pred, return_counts=True))

    # print(np.round(n_y_pred, 2))

    non_weighted_loss = tfa.losses.sigmoid_focal_crossentropy(y_true, y_pred, 0.2, 1.0)
    # print("Non Weighted Loss Sum", np.sum(non_weighted_loss))

    # weighted_loss = tfa.losses.sigmoid_focal_crossentropy(y_true, weighted_predictions)
    # print("Weighted Loss Sum", np.sum(weighted_loss.numpy()))

    return non_weighted_loss

def _generate_test_sample(filter_: str): 
    filter_list = {
        "train": "./data/DNA_test_structures/16x16x16_train.csv",
        "test": "./data/DNA_test_structures/16x16x16_test.csv"
    }

    df = pd.read_csv(filter_list[filter_])
    
    while True: 
        for index, row in df.iterrows(): 

            pdb_code = row["PDB"]

            structure = data.import_pdb(pdb_code)
            neigbour_search, sugar_neigbour_search, phosphate_neigbour_search, base_neigbour_search = _initialise_neighbour_search(structure)

            map_path = os.path.join("data/DNA_test_structures/maps_16", f"{pdb_code}.map")
            if not os.path.isfile(map_path): 
                # print("MAP NOT FOUND")
                continue

            map_ = gemmi.read_ccp4_map(map_path).grid
            map_.normalize()

            translation = [row["X"], row["Y"], row["Z"]]
            # translation = [0,8,40]

            sub_array = np.array(map_.get_subarray(start=translation, shape=[16, 16, 16]))
            output_grid = np.zeros((16, 16, 16, 2))

            for i, x in enumerate(sub_array):
                for j, y in enumerate(x):
                    for k, z in enumerate(y):
                        position = gemmi.Position(translation[0] + i, translation[1] + j, translation[2] + k)

                        any_atom = neigbour_search.find_atoms(position, "\0", radius=params.ns_radius)

                        any_bases = base_neigbour_search.find_atoms(position, "\0", radius=params.ns_radius)
                        any_sugars = sugar_neigbour_search.find_atoms(position, "\0", radius=params.ns_radius)
                        any_phosphate = phosphate_neigbour_search.find_atoms(position, "\0", radius=params.ns_radius)

                        mask = 1 if len(any_atom) > 1 else 0
                        base_mask = 1 if len(any_bases) > 1 else 0
                        sugar_mask = 1 if len(any_sugars) > 1 else 0
                        phosphate_mask = 1 if len(any_phosphate) > 1 else 0

                        if sugar_mask or phosphate_mask or base_mask:
                            # encoded_mask = [0, weights[1]*sugar_mask, weights[2]*phosphate_mask, weights[3]*base_mask]
                            if sugar_mask == 1: 
                                encoded_mask = [0, 1, 0, 0]
                            elif phosphate_mask == 1: 
                                encoded_mask = [0, 0, 1, 0]
                            elif base_mask == 1: 
                                encoded_mask = [0, 0, 0, 1]
                            else:
                                encoded_mask = [0, sugar_mask, phosphate_mask, base_mask]
                            
                            encoded_mask = [0, 1]

                        else:
                            encoded_mask = [1, 0]

                        output_grid[i][j][k] = encoded_mask

            mask = output_grid
            original = sub_array.reshape((16, 16, 16, 1))

            if not np.isfinite(mask).any():
                print(mask)
                print(np.argwhere(np.isnan(mask)))
                continue
            if not np.isfinite(original).any():
                print(original)
                continue

            yield original, mask

def _generate_filtered_sample(filter_: str):
    filter_list = {
        "train": "./data/DNA_test_structures/16x16x16_train.csv",
        "test": "./data/DNA_test_structures/16x16x16_test.csv"
    }

    df = pd.read_csv(filter_list[filter_])
    # df = df.sample(frac=1)
    while True:
        total = len(df)
        yield_count = 0
        
        last_pdb_code = ""
        last_stucture = None
        last_neigbour_search = None
        last_sugar_neigbour_search = None
        last_phosphate_neigbour_search = None
        last_base_neigbour_search = None

        for index, row in df.iterrows():

            pdb_code = row["PDB"]

            if pdb_code == last_pdb_code: 
                structure = last_stucture
                neigbour_search, sugar_neigbour_search, phosphate_neigbour_search, base_neigbour_search = last_neigbour_search, last_sugar_neigbour_search, last_phosphate_neigbour_search, last_base_neigbour_search
            else: 
                structure = data.import_pdb(pdb_code)
                neigbour_search, sugar_neigbour_search, phosphate_neigbour_search, base_neigbour_search = _initialise_neighbour_search(
                structure)
                last_pdb_code = pdb_code
                last_stucture = structure
                last_neigbour_search = neigbour_search
                last_sugar_neigbour_search = sugar_neigbour_search
                last_phosphate_neigbour_search = phosphate_neigbour_search
                last_base_neigbour_search = base_neigbour_search

            map_path = os.path.join("data/DNA_test_structures/maps_16", f"{pdb_code}.map")

            if not os.path.isfile(map_path): 
                continue

            map_ = gemmi.read_ccp4_map(map_path).grid
            map_.normalize()

            translation = [row["X"], row["Y"], row["Z"]]

            sub_array = np.array(map_.get_subarray(start=translation, shape=[16, 16, 16]))
            output_grid = np.zeros((16, 16, 16, 4))

            for i, x in enumerate(sub_array):
                for j, y in enumerate(x):
                    for k, z in enumerate(y):
                        position = gemmi.Position(translation[0] + i, translation[1] + j, translation[2] + k)

                        any_atom = neigbour_search.find_atoms(position, "\0", radius=params.ns_radius)

                        any_bases = base_neigbour_search.find_atoms(position, "\0", radius=params.ns_radius)
                        any_sugars = sugar_neigbour_search.find_atoms(position, "\0", radius=params.ns_radius)
                        any_phosphate = phosphate_neigbour_search.find_atoms(position, "\0", radius=params.ns_radius)

                        mask = 1 if len(any_atom) > 1 else 0
                        base_mask = 1 if len(any_bases) > 1 else 0
                        sugar_mask = 1 if len(any_sugars) > 1 else 0
                        phosphate_mask = 1 if len(any_phosphate) > 1 else 0

                        if sugar_mask or phosphate_mask or base_mask:
                            # encoded_mask = [0, weights[1]*sugar_mask, weights[2]*phosphate_mask, weights[3]*base_mask]
                            if sugar_mask == 1: 
                                encoded_mask = [0, 1, 0, 0]
                            elif phosphate_mask == 1: 
                                encoded_mask = [0, 0, 1, 0]
                            elif base_mask == 1: 
                                encoded_mask = [0, 0, 0, 1]
                            else:
                                encoded_mask = [mask, sugar_mask, phosphate_mask, base_mask]

                        else:
                            encoded_mask = [1, 0, 0, 0]

                        output_grid[i][j][k] = encoded_mask

            mask = output_grid
            original = sub_array.reshape((16, 16, 16, 1))

            if not np.isfinite(mask).any():
                print(mask)
                print(np.argwhere(np.isnan(mask)))
                break
            if not np.isfinite(original).any():
                print(original)
                break
            # print(f"Yielding: {yield_count}/{total}")
            # yield_count += 1

            # print(np.unique(mask, return_counts=True))
            # print(mask)

            yield original, mask


def _initialise_neighbour_search(structure: gemmi.Structure):
    neigbour_search = gemmi.NeighborSearch(structure[0], structure.cell, 1).populate()

    sugar_neigbour_search = gemmi.NeighborSearch(structure[0], structure.cell, 3)
    phosphate_neigbour_search = gemmi.NeighborSearch(structure[0], structure.cell, 3)
    base_neigbour_search = gemmi.NeighborSearch(structure[0], structure.cell, 3)

    sugar_atoms = ["C1'", "C2'", "C3'", "C4'", "C5'", "O2'", "O3'", "O4'", "O5'"]
    phosphate_atoms = ["P", "OP1", "OP2", "O5'", "O3'"]
    base_atoms = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9",
                  "O2", "O4", "O6"]

    for n_ch, chain in enumerate(structure[0]):
        for n_res, res in enumerate(chain):
            for n_atom, atom in enumerate(res):
                if atom.name in sugar_atoms:
                    sugar_neigbour_search.add_atom(atom, n_ch, n_res, n_atom)
                if atom.name in phosphate_atoms:
                    phosphate_neigbour_search.add_atom(atom, n_ch, n_res, n_atom)
                if atom.name in base_atoms:
                    base_neigbour_search.add_atom(atom, n_ch, n_res, n_atom)

    return neigbour_search, sugar_neigbour_search, phosphate_neigbour_search, base_neigbour_search


def test_train(): 

    epochs = 10
    steps = 1000
    batch_size = 1 

    _train_gen = _generate_test_sample("train")
    _test_gen = _generate_test_sample("test")

    input_ = tf.TensorSpec(shape=(16, 16, 16, 1), dtype=tf.float32)
    output = tf.TensorSpec(shape=(16, 16, 16, 2), dtype=tf.int64)

    train_dataset = tf.data.Dataset.from_generator(lambda: _train_gen, output_signature=(input_, output)).repeat(epochs*steps).batch(
        batch_size=batch_size)
    test_dataset = tf.data.Dataset.from_generator(lambda: _test_gen, output_signature=(input_, output)).repeat(epochs*steps).batch(
        batch_size=batch_size)
    model = unet.model_16()
    model.summary()

    # loss = weighted_crossentropy
    loss = weighted_cross_entropy_fn
    # loss = "binary_crossentropy"

    optimiser = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(optimizer=optimiser,
                  loss=loss,
                  metrics=["accuracy", "categorical_accuracy"],
                  # sample_weight_mode='temporal',
                  run_eagerly=True
                  )

    callbacks = [TqdmCallback(verbose=1)]

    model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps,
        validation_data=test_dataset,
        validation_steps=1,
        callbacks=callbacks,
        verbose=0,
    )

    model.save("model_one_sample")


def train():
    #
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(gpus[0], True)

    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    num_threads = 128
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    tf.config.threading.set_inter_op_parallelism_threads(int(num_threads / 2))
    tf.config.threading.set_intra_op_parallelism_threads(int(num_threads / 2))

    # _train_gen = _generate_sample("train")

    _train_gen = _generate_filtered_sample("train")
    _test_gen = _generate_filtered_sample("test")

    batch_size = 8

    input_ = tf.TensorSpec(shape=(16, 16, 16, 1), dtype=tf.float32)
    output = tf.TensorSpec(shape=(16, 16, 16, 4), dtype=tf.int64)

    train_dataset = tf.data.Dataset.from_generator(lambda: _train_gen, output_signature=(input_, output))
    test_dataset = tf.data.Dataset.from_generator(lambda: _test_gen, output_signature=(input_, output))
    model = unet.model_16()
    # model = unet.test_model()
    # model = unet.smaller_model()
    model.summary()

    loss = weighted_crossentropy

    optimiser = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(optimizer=optimiser,
                  loss=loss,
                  metrics=["accuracy", "categorical_accuracy"],
                  # sample_weight_mode='temporal',
                #   run_eagerly=True
                  )

    reduce_lr_on_plat = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, verbose=1,
                                                          mode='auto',
                                                          cooldown=5, min_lr=1e-7)

    weight_path = "multiclass_model_weights_16x16x16.best.hdf5"

    checkpoint = tf.keras.callbacks.ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                                                    save_best_only=True, mode='min', save_weights_only=False)

    train_dataset = train_dataset.repeat(100).batch(
        batch_size=batch_size)

    test_dataset = test_dataset.repeat(100).batch(
        batch_size=batch_size)

    callbacks_list = [checkpoint, reduce_lr_on_plat, TqdmCallback(verbose=1)]
    model.fit(
        train_dataset,
        epochs=10,
        steps_per_epoch=12464,
        validation_data=test_dataset,
        validation_steps=1000,
        callbacks=callbacks_list,
        verbose=0,
    )

    model.save("model_multiclass")

def one_sample_prediction(): 
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    model = tf.keras.models.load_model("model_one_sample", custom_objects={'weighted_crossentropy': weighted_crossentropy})

    _test_gen = _generate_filtered_sample("test")
    test_data = next(_test_gen)
        
    original = test_data[0]
    mask = test_data[1]

    prediction = model.predict(original.reshape(1, 16, 16, 16, 1))

    indices = np.argmax(prediction, axis=-1, keepdims=True)

    print(np.unique(indices, return_counts=True))

def test_prediction():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    _test_gen = _generate_filtered_sample("test")
    weight_path = "multiclass_model_weights.best.hdf5"

    model = unet.model()
    model.load_weights(weight_path)

    test_data = next(_test_gen)
    test_data = next(_test_gen)

    original = test_data[0]
    mask = test_data[1]
    prediction = model.predict(original.reshape(1, 32, 32, 32, 1))

    print(prediction)
    indices = np.argmax(prediction, axis=-1, keepdims=True)

    print(np.unique(indices, return_counts=True))


def evaluate_training_set():
    _train_gen = _generate_filtered_sample("train")
    _test_gen = _generate_filtered_sample("test")

    counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for original, mask in _generate_filtered_sample("train"):
        for dim_1 in mask:
            for dim_2 in dim_1:
                for dim_3 in dim_2:
                    # print(dim_3)
                    index = np.argmax(dim_3, axis=-1)
                    counts[index] += 1
        break
    print(counts)   


def calculate_weights():
    total_weight = np.zeros((4))
    count = 0

    for _, mask in _generate_filtered_sample("train"):
        mask = mask.reshape(32768, 4)
        classes = np.argmax(mask, axis=-1)
        x = class_weight.compute_class_weight(class_weight="balanced", classes=[0, 1, 2, 3], y=classes)
        count += 1
        total_weight += x
        if count > 100:
            break

    average_weights = total_weight / count
    print("Average class weights are ", average_weights)



if __name__ == "__main__":
    params = parameters.Parameters()

    # test_one_hot()
    # calculate_weights()
    # evaluate_training_set()
    # train()
    test_train()
    # one_sample_prediction()
    # test_prediction()

