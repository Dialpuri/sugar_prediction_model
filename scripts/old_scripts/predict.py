import gemmi
import numpy as np
import tensorflow as tf
from focal_loss import BinaryFocalLoss
import import_data as data
from tqdm import tqdm
import train

def map_to_predicted_map(map_path: str):
    model = tf.keras.models._load_model("model_multiclass", custom_objects={'weighted_crossentropy': train.weighted_crossentropy})

    map_ = gemmi.read_ccp4_map(map_path).grid

    a = map_.unit_cell.a
    b = map_.unit_cell.b
    c = map_.unit_cell.c

    overlap = 8

    na = (a // overlap) + 1
    nb = (b // overlap) + 1
    nc = (c // overlap) + 1

    translation_list = []

    for x in range(int(na)):
        for y in range(int(nb)):
            for z in range(int(nc)):
                translation_list.append([x * overlap, y * overlap, z * overlap])

    output_map_a = int(na) * 32
    output_map_b = int(nb) * 32
    output_map_c = int(nc) * 32

    print(output_map_a, output_map_b, output_map_c)

    predicted_map = np.zeros((output_map_a, output_map_b, output_map_c), np.float32)

    model.summary()

    for translation in tqdm(translation_list):
        x, y, z = translation
        sub_array = np.array(map_.get_subarray(start=translation, shape=[32, 32, 32])).reshape(1, 32, 32, 32, 1)
        predicted_sub = model.predict(sub_array)

        arg_max = np.argmax(predicted_sub, axis=4)

        print(np.unique(arg_max, return_counts=True))



        predicted_map[x: x + 32, y: y + 32, z: z + 32] += predicted_sub
        # import pdb; breakpoint()

    import pdb; breakpoint()

    tmp_structure = gemmi.Structure()
    tmp_structure.cell = map_.unit_cell
    tmp_structure.spacegroup_hm = map_.spacegroup.hm

    output_grid = gemmi.FloatGrid()
    output_grid.setup_from(tmp_structure, spacing=map_.spacing[0])

    size_x = predicted_map.shape[0] * map_.spacing[0]
    size_y = predicted_map.shape[1] * map_.spacing[1]
    size_z = predicted_map.shape[2] * map_.spacing[2]
    array_cell = gemmi.UnitCell(size_x, size_y, size_z, 90, 90, 90)

    array_cell = gemmi.UnitCell(size_x, size_y, size_z, 90, 90, 90)
    array_grid = gemmi.FloatGrid(predicted_map, array_cell)

    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = array_grid
    ccp4.update_ccp4_header()

    output_path = map_path.replace("external_test_maps", "output_maps")
    output_path = output_path.replace("mtz", "mrc")
    ccp4.write_ccp4_map(output_path)


if __name__ == "__main__":
    mtz = gemmi.read_mtz_file("data/DNA_test_structures/external_test_maps/1hr2.mtz")
    map = mtz.transform_f_phi_to_map("FWT", "PHWT")

    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = map
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map("data/DNA_test_structures/external_test_maps/1hr2.map")
    map_to_predicted_map("data/DNA_test_structures/external_test_maps/1hr2.map")
