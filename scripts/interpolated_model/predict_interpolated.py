import gemmi
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm

def get_bounding_box(grid: gemmi.FloatGrid) -> gemmi.PositionBox:
    extent = gemmi.find_asu_brick(grid.spacegroup).get_extent()
    corners = [
        grid.unit_cell.orthogonalize(fractional)
        for fractional in (
            extent.minimum,
            gemmi.Fractional(extent.maximum[0], extent.minimum[1], extent.minimum[2]),
            gemmi.Fractional(extent.minimum[0], extent.maximum[1], extent.minimum[2]),
            gemmi.Fractional(extent.minimum[0], extent.minimum[1], extent.maximum[2]),
            gemmi.Fractional(extent.maximum[0], extent.maximum[1], extent.minimum[2]),
            gemmi.Fractional(extent.maximum[0], extent.minimum[1], extent.maximum[2]),
            gemmi.Fractional(extent.minimum[0], extent.maximum[1], extent.maximum[2]),
            extent.maximum,
        )
    ]
    min_x = min(corner[0] for corner in corners)
    min_y = min(corner[1] for corner in corners)
    min_z = min(corner[2] for corner in corners)
    max_x = max(corner[0] for corner in corners)
    max_y = max(corner[1] for corner in corners)
    max_z = max(corner[2] for corner in corners)
    box = gemmi.PositionBox()
    box.minimum = gemmi.Position(min_x, min_y, min_z)
    box.maximum = gemmi.Position(max_x, max_y, max_z)
    return box



def map_to_predicted_map(map_path: str):
    model = tf.keras.models.load_model("models/interpolated_model_2", custom_objects={'sigmoid_focal_crossentropy': tfa.losses.sigmoid_focal_crossentropy})

    map_ = gemmi.read_ccp4_map(map_path).grid

    grid_spacing = 0.7

    map_.normalize()
    
    box = get_bounding_box(map_)
    size = box.get_size()
    num_x = -(-int(size.x / grid_spacing) // 16 * 16)
    num_y = -(-int(size.y / grid_spacing) // 16 * 16)
    num_z = -(-int(size.z / grid_spacing) // 16 * 16)
    array = np.zeros((num_x, num_y, num_z), dtype=np.float32)
    scale = gemmi.Mat33(
        [[grid_spacing, 0, 0], [0, grid_spacing, 0], [0, 0, grid_spacing]]
    )
    transform = gemmi.Transform(scale, box.minimum)
    map_.interpolate_values(array, transform)
    cell = gemmi.UnitCell(size.x, size.y, size.z, 90, 90, 90)
    map_array = gemmi.FloatGrid(array, cell)

    a = map_array.unit_cell.a
    b = map_array.unit_cell.b
    c = map_array.unit_cell.c

    overlap = 32

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

    predicted_map = np.zeros((output_map_a, output_map_b, output_map_c), np.float32)

    for translation in tqdm(translation_list):
        x, y, z = translation
        sub_array = np.array(map_array.get_subarray(start=translation, shape=[32, 32, 32])).reshape(1, 32, 32, 32, 1)
        predicted_sub = model.predict(sub_array).squeeze()
        arg_max = np.argmax(predicted_sub, axis=-1)
        predicted_map[x: x + 32, y: y + 32, z: z + 32] += arg_max

    np.save("./data/DNA_test_structures/output_maps/predicted_map.npy", predicted_map)

    size_x = predicted_map.shape[0] * map_array.spacing[0]
    size_y = predicted_map.shape[1] * map_array.spacing[1]
    size_z = predicted_map.shape[2] * map_array.spacing[2]
    array_cell = gemmi.UnitCell(size_x, size_y, size_z, 90, 90, 90)

    array_cell = gemmi.UnitCell(size_x, size_y, size_z, 90, 90, 90)
    array_grid = gemmi.FloatGrid(predicted_map, array_cell)

    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = array_grid
    ccp4.update_ccp4_header()

    output_path = map_path.replace("external_test_maps", "output_maps")
    output_path = output_path.replace("mtz", "mrc")
    ccp4.write_ccp4_map(output_path)

    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = map_array
    ccp4.update_ccp4_header()

    output_path = map_path.replace("external_test_maps", "interpolated_maps")
    output_path = output_path.replace("mtz", "mrc")
    ccp4.write_ccp4_map(output_path)


def load_predicted_map(): 
    array = np.load("./data/DNA_test_structures/output_maps/predicted_map.npy")
    print(array.shape)


if __name__ == "__main__":
    # mtz = gemmi.read_mtz_file("data/DNA_test_structures/external_test_maps/1hr2.mtz")
    # map = mtz.transform_f_phi_to_map("FWT", "PHWT")
    # ccp4 = gemmi.Ccp4Map()
    # ccp4.grid = map
    # ccp4.update_ccp4_header()
    # ccp4.write_ccp4_map("data/DNA_test_structures/external_test_maps/1hr2.map")

    map_to_predicted_map("data/DNA_test_structures/external_test_maps/1hr2.map")
    # load_predicted_map()