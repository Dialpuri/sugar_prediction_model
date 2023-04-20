import gemmi
import numpy as np
import matplotlib.pyplot as plt
import import_data as data
import os
from typing import Tuple, List
from tqdm import tqdm
from multiprocessing import Pool
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
import random


@dataclass
class Names:
    sugar_file: str = "_interpolated_sugar"
    phosphate_file: str = "_interpolated_phosphate"
    base_file: str = "_interpolated_base"
    no_sugar_file: str = "_interpolated_no_sugar"
    density_file: str = "_interpolated_density"


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


def plot_interpolated_values():
    map_path = "./data/DNA_test_structures/maps_16/1ais.map"

    map_ = gemmi.read_ccp4_map(map_path).grid
    map_.normalize()

    array, box_min = _interpolate_input_grid(map_)

    x = np.arange(array.shape[0])
    y = np.arange(array.shape[1])
    z = np.arange(array.shape[2])

    X, Y, Z = np.meshgrid(x, y, z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    flat_array = array.flatten()
    ax.scatter3D(X, Y, Z, s=flat_array, c=flat_array, alpha=0.5)
    plt.savefig("./output/1ais_interp.png")


def _initialise_neighbour_search(structure: gemmi.Structure, radius: int = 3):
    neigbour_search = gemmi.NeighborSearch(structure[0], structure.cell, 1).populate()

    sugar_neigbour_search = gemmi.NeighborSearch(structure[0], structure.cell, radius)
    phosphate_neigbour_search = gemmi.NeighborSearch(structure[0], structure.cell, radius)
    base_neigbour_search = gemmi.NeighborSearch(structure[0], structure.cell, radius)

    sugar_atoms = ["C1'", "C2'", "C3'", "C4'", "C5'", "O2'", "O3'", "O4'", "O5'"]
    phosphate_atoms = ["P", "OP1", "OP2", "O5'", "O3'"]
    base_atoms = [
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "N1",
        "N2",
        "N3",
        "N4",
        "N5",
        "N6",
        "N7",
        "N8",
        "N9",
        "O2",
        "O4",
        "O6",
    ]

    for n_ch, chain in enumerate(structure[0]):
        for n_res, res in enumerate(chain):
            for n_atom, atom in enumerate(res):
                if atom.name in sugar_atoms:
                    sugar_neigbour_search.add_atom(atom, n_ch, n_res, n_atom)
                if atom.name in phosphate_atoms:
                    phosphate_neigbour_search.add_atom(atom, n_ch, n_res, n_atom)
                if atom.name in base_atoms:
                    base_neigbour_search.add_atom(atom, n_ch, n_res, n_atom)

    return (
        neigbour_search,
        sugar_neigbour_search,
        phosphate_neigbour_search,
        base_neigbour_search,
    )


def generate_c_alpha_positions(map_path: str, pdb_code: str, sample_size: int ): 


    # Need to find positions to add to the help file which will include position of high density but no sugars

    grid_spacing = 0.7

    input_grid = gemmi.read_ccp4_map(map_path).grid
    input_grid.normalize()
    try:
        structure = data.import_pdb(pdb_code)
    except FileNotFoundError:
        print("[FAILED]:", map_path, pdb_code)
        return

    box = get_bounding_box(input_grid)
    size = box.get_size()
    num_x = -(-int(size.x / grid_spacing) // 16 * 16)
    num_y = -(-int(size.y / grid_spacing) // 16 * 16)
    num_z = -(-int(size.z / grid_spacing) // 16 * 16)
    array = np.zeros((num_x, num_y, num_z), dtype=np.float32)
    scale = gemmi.Mat33(
        [[grid_spacing, 0, 0], [0, grid_spacing, 0], [0, 0, grid_spacing]]
    )
    transform = gemmi.Transform(scale, box.minimum)
    input_grid.interpolate_values(array, transform)

    cell = gemmi.UnitCell(size.x, size.y, size.z, 90, 90, 90)

    c_alpha_search = gemmi.NeighborSearch(structure[0], structure.cell, 3)

    c_alpha_atoms = ["CA", "CB"]

    grid_sample_size = 32

    for n_ch, chain in enumerate(structure[0]):
            for n_res, res in enumerate(chain):
                for n_atom, atom in enumerate(res):
                    if atom.name in c_alpha_atoms:
                        c_alpha_search.add_atom(atom, n_ch, n_res, n_atom)

    potential_positions = []

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for k in range(array.shape[2]):
                index_pos = gemmi.Vec3(i, j, k)
                position = gemmi.Position(transform.apply(index_pos))

                any_protein_backbone = c_alpha_search.find_atoms(position, "\0", radius=3)

                if len(any_protein_backbone) > 0: 
                    translatable_position = (i-grid_sample_size/2, j-grid_sample_size/2, k-grid_sample_size/2)
                    potential_positions.append(translatable_position)

    if len(potential_positions) != 0: 
        return random.sample(potential_positions, sample_size)
    return []

def generate_class_files(map_path: str, pdb_code: str, base_dir: str, radius: int = 3):
    grid_spacing = 0.7

    input_grid = gemmi.read_ccp4_map(map_path).grid
    input_grid.normalize()
    try:
        structure = data.import_pdb(pdb_code)
    except FileNotFoundError:
        print("[FAILED]:", map_path, pdb_code)
        return

    box = get_bounding_box(input_grid)
    size = box.get_size()
    num_x = -(-int(size.x / grid_spacing) // 16 * 16)
    num_y = -(-int(size.y / grid_spacing) // 16 * 16)
    num_z = -(-int(size.z / grid_spacing) // 16 * 16)
    array = np.zeros((num_x, num_y, num_z), dtype=np.float32)
    scale = gemmi.Mat33(
        [[grid_spacing, 0, 0], [0, grid_spacing, 0], [0, 0, grid_spacing]]
    )
    transform = gemmi.Transform(scale, box.minimum)
    input_grid.interpolate_values(array, transform)

    cell = gemmi.UnitCell(size.x, size.y, size.z, 90, 90, 90)

    (
        neigbour_search,
        sugar_neigbour_search,
        phosphate_neigbour_search,
        base_neigbour_search,
    ) = _initialise_neighbour_search(structure)

    no_sugar_map = np.zeros(array.shape, dtype=np.float32)
    sugar_map = np.zeros(array.shape, dtype=np.float32)
    phosphate_map = np.zeros(array.shape, dtype=np.float32)
    base_map = np.zeros(array.shape, dtype=np.float32)

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for k in range(array.shape[2]):
                # indices are i,j,k
                # need to turn those indicies into xyz from the transformed
                index_pos = gemmi.Vec3(i, j, k)
                position = gemmi.Position(transform.apply(index_pos))

                any_bases = base_neigbour_search.find_atoms(position, "\0", radius=radius)
                any_sugars = sugar_neigbour_search.find_atoms(position, "\0", radius=radius)
                any_phosphate = phosphate_neigbour_search.find_atoms(
                    position, "\0", radius=radius
                )

                base_mask = 1.0 if len(any_bases) > 1 else 0.0
                sugar_mask = 1.0 if len(any_sugars) > 1 else 0.0
                phosphate_mask = 1.0 if len(any_phosphate) > 1 else 0.0

                if sum([base_mask, sugar_mask, phosphate_mask]) == 0:
                    no_sugar_map[i][j][k] = 1.0
                else:
                    no_sugar_map[i][j][k] = 0.0

                sugar_map[i][j][k] = sugar_mask
                phosphate_map[i][j][k] = phosphate_mask
                base_map[i][j][k] = base_mask

    output_dir = os.path.join(base_dir, pdb_code)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    no_sugar_grid = gemmi.FloatGrid(no_sugar_map)
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = no_sugar_grid
    ccp4.grid.unit_cell.set(array.shape[0], array.shape[1], array.shape[2], 90, 90, 90)
    ccp4.grid.spacegroup = gemmi.SpaceGroup("P1")
    ccp4.update_ccp4_header()
    density_path = os.path.join(output_dir, f"{pdb_code}{names.no_sugar_file}.map")
    ccp4.write_ccp4_map(density_path)

    density_grid = gemmi.FloatGrid(array)
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = density_grid
    ccp4.grid.unit_cell.set(array.shape[0], array.shape[1], array.shape[2], 90, 90, 90)
    ccp4.grid.spacegroup = gemmi.SpaceGroup("P1")
    ccp4.update_ccp4_header()
    density_path = os.path.join(output_dir, f"{pdb_code}{names.density_file}.map")
    ccp4.write_ccp4_map(density_path)

    sugar_grid = gemmi.FloatGrid(sugar_map)
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = sugar_grid
    ccp4.grid.unit_cell.set(array.shape[0], array.shape[1], array.shape[2], 90, 90, 90)
    ccp4.grid.spacegroup = gemmi.SpaceGroup("P1")
    ccp4.update_ccp4_header()
    sugar_path = os.path.join(output_dir, f"{pdb_code}{names.sugar_file}.map")
    ccp4.write_ccp4_map(sugar_path)

    phosphate_grid = gemmi.FloatGrid(phosphate_map)
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = phosphate_grid
    ccp4.grid.unit_cell.set(array.shape[0], array.shape[1], array.shape[2], 90, 90, 90)
    ccp4.grid.spacegroup = gemmi.SpaceGroup("P1")
    ccp4.update_ccp4_header()
    phosphate_path = os.path.join(output_dir, f"{pdb_code}{names.phosphate_file}.map")
    ccp4.write_ccp4_map(phosphate_path)

    base_grid = gemmi.FloatGrid(base_map)
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = base_grid
    ccp4.grid.unit_cell.set(array.shape[0], array.shape[1], array.shape[2], 90, 90, 90)
    ccp4.grid.spacegroup = gemmi.SpaceGroup("P1")
    ccp4.update_ccp4_header()
    base_path = os.path.join(output_dir, f"{pdb_code}{names.base_file}.map")
    ccp4.write_ccp4_map(base_path)


def get_map_list(directory: str) -> List[Tuple[str, str]]:
    # Returns list of tuples containing map_path and pdb_code
    map_list = os.listdir(directory)
    return [
        (os.path.join(directory, map_file), map_file.replace(".map", ""))
        for map_file in map_list
        if ".map" in map_file
    ]


def get_data_dirs(base_dir: str) -> List[Tuple[str, str]]:
    data_dirs = os.listdir(base_dir)
    return [
        (os.path.join(base_dir, directory), directory)
        for directory in data_dirs
        if os.path.isfile(
            os.path.join(base_dir, directory, f"{directory}{names.density_file}.map")
        )
    ]


def map_worker(data: Tuple[str, str]):
    output_dir = "./low_res_dataset"
    map_file, pdb_code = data
    generate_class_files(map_file, pdb_code, output_dir, radius=1.5)


def generate_map_files():
    map_list = get_map_list("data/low_res_data/map")

    with Pool() as pool:
        r = list(tqdm(pool.imap(map_worker, map_list), total=len(map_list)))


def generate_candidate_position_list(
    base_dir: str, pdb_code: str, threshold: float
) -> List[List[int]]:
    sugar_map = os.path.join(base_dir, f"{pdb_code}{names.sugar_file}.map")
    input_grid = gemmi.read_ccp4_map(sugar_map).grid

    a = input_grid.unit_cell.a
    b = input_grid.unit_cell.b
    c = input_grid.unit_cell.c

    overlap = 16

    box_dimensions = [32, 32, 32]
    total_points = box_dimensions[0] ** 3

    na = (a // overlap) + 1
    nb = (b // overlap) + 1
    nc = (c // overlap) + 1

    translation_list = []

    for x in range(int(na)):
        for y in range(int(nb)):
            for z in range(int(nc)):
                translation_list.append([x * overlap, y * overlap, z * overlap])

    candidate_translations = []

    for translation in translation_list:
        sub_array = np.array(
            input_grid.get_subarray(start=translation, shape=box_dimensions)
        )

        sum = np.sum(sub_array)
        if (sum / total_points) > threshold:
            candidate_translations.append(translation)

    print(len(candidate_translations))

    return candidate_translations


def help_file_worker(data_tuple: Tuple[str, str]):
    base_dir, pdb_code = data_tuple

    candidate_translations = generate_candidate_position_list(base_dir, pdb_code, 0.03)

    help_file_path = os.path.join(base_dir, "validated_translations.csv")

    with open(help_file_path, "w") as help_file:
        help_file.write("X,Y,Z\n")

        for translation in candidate_translations:
            help_file.write(f"{translation[0]},{translation[1]},{translation[2]}\n")


def generate_help_files():
    # Must be run after map files have been generated

    data_directories = get_data_dirs("./low_res_dataset")

    with Pool() as pool:
        r = list(
            tqdm(
                pool.imap(help_file_worker, data_directories),
                total=len(data_directories),
            )
        )


def combine_help_files():
    base_dir = "./low_res_dataset"

    main_df = pd.DataFrame(columns=["PDB", "X", "Y", "Z"])

    for dir in os.scandir(base_dir):
        context_path = os.path.join(dir.path, "validated_translations_calpha.csv")

        df = pd.read_csv(context_path)

        df = df.assign(PDB=dir.name)

        main_df = pd.concat([main_df, df])

    print(main_df)

    main_df.to_csv("./data/low_res_data/combined_dataset.csv", index=False)


def generate_test_train_split():

    df = pd.read_csv("./data/low_res_data/combined_dataset.csv")

    train, test = train_test_split(df, test_size=0.2)

    train.to_csv("./data/low_res_data/test.csv", index=False)
    test.to_csv("./data/low_res_data/train.csv", index=False)


def seeder(data: Tuple[str, str]): 
    output_dir = "./low_res_dataset"
    map_file, pdb_code = data

    pdb_folder = os.path.join(output_dir, pdb_code)
    output_path = os.path.join(pdb_folder, "validated_translations_calpha.csv")

    if os.path.isfile(output_path):
        return 

    validated_translation_file = os.path.join(pdb_folder, "validated_translations.csv")

    if not os.path.isfile(validated_translation_file):
        return

    df = pd.read_csv(validated_translation_file)

    if len(df) < 10: 
        sample_size = 4
    else:
        sample_size = len(df) // 5

    samples = generate_c_alpha_positions(map_path=map_file, pdb_code=pdb_code, sample_size=sample_size)

    output_df = pd.concat([df, pd.DataFrame(samples, columns=["X","Y","Z"])])
    output_df.to_csv(output_path, index=False)

def seed_c_alpha_positions(): 
    map_list = get_map_list("./data/low_res_data/map")

    with Pool() as pool:
        r = list(tqdm(pool.imap(seeder, map_list), total=len(map_list)))

def main():

    # generate_map_files()
    # seed_c_alpha_positions()
    # combine_help_files()
    # generate_test_train_split()
    # generate_c_alpha_positions("data/DNA_test_structures/maps_16/1azp.map", "1azp", "./data/DNA_test_structures")
    # generate_class_files("./data/DNA_test_structures/external_test_maps/1hr2.map", "1hr2", "./data/DNA_test_structures")
    # generate_test_train_split()
    # generate_map_files()
    # generate_help_files()
    seed_c_alpha_positions()
    combine_help_files()
    generate_test_train_split()



if __name__ == "__main__":
    names = Names()
    main()
