from multiprocessing import Pool

import parameters
import numpy as np
import import_data as data
import os
import gemmi
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split


def get_map_list(filter_: str) -> list[str]:
    return [path.path for path in os.scandir(params.maps_dir) if filter_ in path.name]


def get_output(map_path):
    pdb_code = map_path.split("/")[-1].split(".")[0].strip("test").strip("_")

    structure = data.import_pdb(pdb_code)
    neigbour_search = gemmi.NeighborSearch(structure[0], structure.cell, 1).populate()

    sugar_neigbour_search = gemmi.NeighborSearch(structure[0], structure.cell, 3)
    phosphate_neigbour_search = gemmi.NeighborSearch(structure[0], structure.cell, 3)
    base_neigbour_search = gemmi.NeighborSearch(structure[0], structure.cell, 3)

    sugar_atoms = ["C1'", "C2'", "C3'", "C4'", "C5'", "O2'", "O3'", "O4'", "O5'"]
    phosphate_atoms = ["P", "OP1", "OP2", "O5'", "O3'"]
    base_atoms = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8",
                  "N9",
                  "O2", "O4", "O6"
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

    map_ = gemmi.read_ccp4_map(map_path).grid

    map_.normalize()

    a = map_.unit_cell.a
    b = map_.unit_cell.b
    c = map_.unit_cell.c

    # print("Unit cell dimensions ", a, b, c)

    overlap = 8

    na = (a // overlap) + 1
    nb = (b // overlap) + 1
    nc = (c // overlap) + 1

    translation_list = []

    for x in range(int(na)):
        for y in range(int(nb)):
            for z in range(int(nc)):
                translation_list.append([x * overlap, y * overlap, z * overlap])

    output_list = []

    box_size = 16

    for translation in translation_list:
        sub_array = np.array(map_.get_subarray(start=translation, shape=[box_size, box_size, box_size]))
        output_grid = np.zeros((box_size, box_size, box_size, 4))

        for i, x in enumerate(sub_array):
            for j, y in enumerate(x):
                for k, z in enumerate(y):
                    position = gemmi.Position(translation[0] + i, translation[1] + j, translation[2] + k)

                    # print(translation[0]+i, translation[1]+j, translation[2]+k)

                    any_atom = neigbour_search.find_atoms(position, "\0", radius=params.ns_radius)

                    any_bases = base_neigbour_search.find_atoms(position, "\0", radius=params.ns_radius)
                    any_sugars = sugar_neigbour_search.find_atoms(position, "\0", radius=params.ns_radius)
                    any_phosphate = phosphate_neigbour_search.find_atoms(position, "\0", radius=params.ns_radius)

                    mask = 1 if len(any_atom) > 1 else 0
                    base_mask = 1 if len(any_bases) > 1 else 0
                    sugar_mask = 1 if len(any_sugars) > 1 else 0
                    phosphate_mask = 1 if len(any_phosphate) > 1 else 0

                    # encoded_mask = tf.one_hot(mask, depth=2)

                    encoded_mask = [mask, sugar_mask, phosphate_mask, base_mask]

                    output_grid[i][j][k] = encoded_mask

        mask = output_grid.reshape((box_size, box_size, box_size, 4))

        # yield (mask == 1).sum(), translation
        output_list.append((pdb_code, np.unique(output_grid, return_counts=True)[1], translation))
    return output_list


def worker(map_path):
    output_list = get_output(map_path)

    with open(params.precompute_list_dir, "a") as output_file:

        for pdb_code, unique, translation in output_list:
            if len(unique) > 1:
                output_line = f"{pdb_code},{translation[0]},{translation[1]},{translation[2]},{unique[0]},{unique[1]}\n"
            else:
                output_line = f"{pdb_code},{translation[0]},{translation[1]},{translation[2]},{unique[0]},\n"

            output_file.write(output_line)

def compute_all():
    map_ = get_map_list("test")

    with open(params.precompute_list_dir, "w") as file_:
        file_.write("PDB,X,Y,Z,0,1\n")

    with Pool() as pool:
        r = list(tqdm(pool.imap(worker, iterable=map_), total=len(map_)))


def filter_precomputed():
    df = pd.read_csv(params.precompute_list_dir)

    filtered_output = "./data/DNA_test_structures/16x16x16_filtered.csv"

    df = df.dropna()

    filtered_df = df[df["1"] > 3_000]
    # print(filtered_df)
    filtered_df.to_csv(filtered_output, index=False)
    #
    # print(over_5_000)

def generate_test_train_split(): 
    df = pd.read_csv("./data/DNA_test_structures/16x16x16_filtered.csv")
    train, test = train_test_split(df, test_size=0.2)

    output_base = "./data/DNA_test_structures"
    train_file = os.path.join(output_base, "16x16x16_train.csv")
    test_file = os.path.join(output_base, "16x16x16_test.csv")

    train.to_csv(train_file, index=False)
    test.to_csv(test_file, index=False)


if __name__ == "__main__":

    params = parameters.Parameters()

    # filter_precomputed()
    generate_test_train_split()
    # compute_all()
