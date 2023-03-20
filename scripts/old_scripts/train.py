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
matplotlib.use('Agg', force=True)
from tqdm.keras import TqdmCallback


def get_map_list(filter_: str) -> list[str]:
    return [path.path for path in os.scandir(params.maps_dir) if filter_ in path.name]


def weighted_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0.0, axis=-1):
    weights = {
        # 0: 0.34070462,
        0: 0.01,
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
    print(np.round(n_y_pred, 2))


    non_weighted_loss = tfa.losses.sigmoid_focal_crossentropy(y_true, y_pred).numpy()
    print("Non Weighted Loss Sum", np.sum(non_weighted_loss))

    weighted_loss = tfa.losses.sigmoid_focal_crossentropy(y_true, weighted_predictions)
    print("Weighted Loss Sum", np.sum(weighted_loss.numpy()))

    return weighted_loss


    # label_smoothing = tf.convert_to_tensor(label_smoothing, dtype=y_pred.dtype)
    #
    # def _smooth_labels():
    #     num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)
    #     return y_true * (1.0 - label_smoothing) + (
    #             label_smoothing / num_classes
    #     )
    #
    # y_true = tf.__internal__.smart_cond.smart_cond(
    #     label_smoothing, _smooth_labels, lambda: y_true
    # )

    # print(tf.keras.backend.get_value(tf.keras.backend.categorical_crossentropy(
    #         y_true, y_pred, from_logits=from_logits, axis=axis
    # ))[0][0])
    #
    #
    # print(tf.keras.backend.get_value(tf.keras.backend.categorical_crossentropy(
    #         y_true, weighted_predictions, from_logits=from_logits, axis=axis
    # ))[0][0])


        # print(tf.keras.backend.categorical_crossentropy(
        #     y_true, y_pred, from_logits=from_logits, axis=axis
        # ).eval())
    #
    # return tf.keras.backend.categorical_crossentropy(
    #     y_true, weighted_predictions, from_logits=from_logits, axis=axis
    # )


def _generate_filtered_sample(filter_: str):
    filter_list = {
        "train": "./data/DNA_test_structures/precalc_10000.csv",
        "test": "./data/DNA_test_structures/precalc_all_test_1000.csv"
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

            map_path = os.path.join("data/DNA_test_structures/maps", f"{pdb_code}_{filter_}.map")
            map_ = gemmi.read_ccp4_map(map_path).grid
            map_.normalize()

            translation = [row["X"], row["Y"], row["Z"]]

            sub_array = np.array(map_.get_subarray(start=translation, shape=[32, 32, 32]))
            output_grid = np.zeros((32, 32, 32, 4))

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

                        if mask == 1:
                            # encoded_mask = [0, weights[1]*sugar_mask, weights[2]*phosphate_mask, weights[3]*base_mask]
                            if sugar_mask == 1: 
                                encoded_mask = [0, 1, 0, 0]
                            elif phosphate_mask == 1: 
                                encoded_mask = [0, 0, 1, 0]
                            elif base_mask == 1: 
                                encoded_mask = [0, 0, 0, 1]
                            
                        else:
                            encoded_mask = [1, 0, 0, 0]

                        output_grid[i][j][k] = encoded_mask

            mask = output_grid
            original = sub_array.reshape((32, 32, 32, 1))

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

    input = tf.TensorSpec(shape=(32, 32, 32, 1), dtype=tf.float32)
    output = tf.TensorSpec(shape=(32, 32, 32, 4), dtype=tf.int64)

    train_dataset = tf.data.Dataset.from_generator(lambda: _train_gen, output_signature=(input, output))
    test_dataset = tf.data.Dataset.from_generator(lambda: _test_gen, output_signature=(input, output))
    model = unet.model()
    # model = unet.test_model()
    # model = unet.smaller_model()
    model.summary()
    #
    # loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=2,
    #                                                from_logits=False)

    # loss = BinaryFocalLoss(gamma=2)
    # loss = tf.keras.losses.CategoricalCrossentropy()

    loss = weighted_crossentropy

    # loss = tfa.losses.sigmoid_focal_crossentropy()

    optimiser = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(optimizer=optimiser,
                  loss=loss,
                  metrics=["accuracy", "categorical_accuracy"],
                  # sample_weight_mode='temporal',
                  run_eagerly=True
                  )

    reduce_lr_on_plat = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, verbose=1,
                                                          mode='auto',
                                                          cooldown=5, min_lr=1e-7)

    weight_path = "multiclass_model_weights.best.hdf5"

    checkpoint = tf.keras.callbacks.ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                                                    save_best_only=True, mode='min', save_weights_only=False)

    # train_dataset = train_dataset.cache("cached_data")
    train_dataset = train_dataset.repeat(100).batch(
        batch_size=batch_size)

    test_dataset = test_dataset.repeat(100).batch(
        batch_size=batch_size)

    #           NO ATOMS      SUGAR     PHOSPHATE      BASE
    weights = {
        0: 0.34070462,
        1: 2.15107971,
        2: 10.56569157,
        3: 2.10228326
    }

    callbacks_list = [checkpoint, reduce_lr_on_plat, TqdmCallback(verbose=1)]
    model.fit(
        train_dataset,
        epochs=10,
        steps_per_epoch=15802,
        validation_data=test_dataset,
        validation_steps=1000,
        callbacks=callbacks_list,
        verbose=0,
    )

    model.save("model_multiclass")


def test_prediction():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    _test_gen = _generate_filtered_sample("test")
    # model = tf.keras.models.load_model("model_multiclass", custom_objects={'weighted_crossentropy': weighted_crossentropy})
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

    # prediction = prediction.squeeze()
    # decoded = tf.argmax(prediction, axis=-1)
    # decoded_mask = np.array(tf.argmax(mask, axis=-1)).flatten()
    #
    # x = np.indices(decoded.shape)[0]
    # y = np.indices(decoded.shape)[1]
    # z = np.indices(decoded.shape)[2]
    # col = np.array(decoded).flatten()
    #
    # fig = plt.figure()
    # ax3D = fig.add_subplot(1, 2, 1, projection='3d')
    # ax3D2 = fig.add_subplot(1, 2, 2, projection='3d')
    #
    # p3d = ax3D.scatter(x, y, z, s=col, alpha=0.4)
    #
    # p3d = ax3D2.scatter(x, y, z, s=decoded_mask, alpha=0.4)
    # ax3D.set_title("Predicted")
    # ax3D2.set_title("Ground Truth")
    # plt.savefig("./debug/prediction_32x32x32_multiclass.png")
    # plt.show()


def evaluate_training_set():
    _train_gen = _generate_filtered_sample("train")
    _test_gen = _generate_filtered_sample("test")

    # start = time.time()
    # training_length = len([g for g in _train_gen])
    # print("Training gen length = ", )
    # end = time.time()
    # print("Training generation took\n Total:", end-start, "Per Step: ", (end-start)/training_length )

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

    # start = time.time()
    # for x in _train_gen:
    #     now = time.time()
    #     print(x[0].shape, x[1].shape, now-start)
    #     start = now

    #
    # for train_data in _train_gen:
    #     density = train_data[0]
    #     mask = train_data[1]
    #
    #     #
    #     x = np.indices(density.shape)[0]
    #     y = np.indices(density.shape)[1]
    #     z = np.indices(density.shape)[2]
    #
    #     colours = []
    #     sizes = []
    #     for i in range(0, 32):
    #         for j in range(0, 32):
    #             for k in range(0, 32):
    #                 value = mask[i][j][k]
    #
    #                 if np.array_equal(value, [0, 0, 0]):
    #                     colours.append("black")
    #                     sizes.append(0)
    #                 elif np.array_equal(value, [1, 0, 0]):
    #                     colours.append("red")
    #                     sizes.append(1)
    #
    #                 elif np.array_equal(value, [0, 1, 0]):
    #                     colours.append("green")
    #                     sizes.append(1)
    #
    #                 elif np.array_equal(value, [0, 0, 1]):
    #                     colours.append("blue")
    #                     sizes.append(1)
    #
    #                 elif np.array_equal(value, [0, 1, 1]):
    #                     colours.append("orange")
    #                     sizes.append(3)
    #
    #                 elif np.array_equal(value, [1, 1, 0]):
    #                     colours.append("purple")
    #                     sizes.append(3)
    #
    #                 elif np.array_equal(value, [1, 1, 1]):
    #                     colours.append("brown")
    #                     sizes.append(3)
    #
    #                 else:
    #                     colours.append("yellow")
    #                     sizes.append(3)
    #
    #     fig = plt.figure()
    #     ax3D = fig.add_subplot(projection='3d')
    #     p3d = ax3D.scatter(x, y, z, c=colours, alpha=0.4, s=sizes)
    #
    #     import matplotlib.patches as mpatches
    #     handles, labels = ax3D.get_legend_handles_labels()
    #     patch1 = mpatches.Patch(color='black', label='No atoms')
    #     patch2 = mpatches.Patch(color='red', label='Sugar nearby')
    #     patch3 = mpatches.Patch(color='green', label='Phosphate nearby')
    #     patch4 = mpatches.Patch(color='blue', label='Base nearby')
    #     patch5 = mpatches.Patch(color='yellow', label='Other')
    #     patch6 = mpatches.Patch(color='orange', label='Phosphate + Base')
    #     patch7 = mpatches.Patch(color='purple', label='Sugar + Phosphate')
    #     patch8 = mpatches.Patch(color='brown', label='All nearby')
    #
    #     handles.append(patch1)
    #     handles.append(patch2)
    #     handles.append(patch3)
    #     handles.append(patch4)
    #     handles.append(patch5)
    #     handles.append(patch6)
    #     handles.append(patch7)
    #     handles.append(patch8)
    #     plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1.03, 0.5), fontsize=7)
    #     plt.show()
    #     plt.tight_layout()
    #     fig.subplots_adjust(right=0.8)
    #
    #     plt.savefig("./debug/test_3channel_pred.png")
    #
    #     # unique, counts = np.unique(mask, return_counts=True)
    #     # occurrence_dict = dict(zip(unique, counts))
    #     #
    #     # print(occurrence_dict)
    #     break
    #


def test_one_hot():
    indices = [1, 1, 0]
    print(tf.one_hot(indices, depth=2))


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
    #    evaluate_training_set()
    train()
    # test_prediction()
