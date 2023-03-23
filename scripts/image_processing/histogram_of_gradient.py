from multiprocessing import Pool
import os
from typing import Tuple
import time
import gemmi 
import sys

from tqdm import tqdm
sys.path.append("/home/jordan/dev/sugar_position_pred/scripts/interpolated_model")
import utils
import import_data as data
import math 
import matplotlib.pyplot as plt
import numpy as np 


def _initialise_neighbour_search(structure: gemmi.Structure, radius: int = 3):
    neigbour_search = gemmi.NeighborSearch(structure[0], structure.cell, 1).populate()

    sugar_neigbour_search = gemmi.NeighborSearch(structure[0], structure.cell, radius)
    phosphate_neigbour_search = gemmi.NeighborSearch(structure[0], structure.cell, radius)
    base_neigbour_search = gemmi.NeighborSearch(structure[0], structure.cell, radius)
    protein_neigbour_search = gemmi.NeighborSearch(structure[0], structure.cell, radius)

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

    protein_backbone = ["CA", "CB"]

    for n_ch, chain in enumerate(structure[0]):
        for n_res, res in enumerate(chain):
            for n_atom, atom in enumerate(res):
                if atom.name in sugar_atoms:
                    sugar_neigbour_search.add_atom(atom, n_ch, n_res, n_atom)
                if atom.name in phosphate_atoms:
                    phosphate_neigbour_search.add_atom(atom, n_ch, n_res, n_atom)
                if atom.name in base_atoms:
                    base_neigbour_search.add_atom(atom, n_ch, n_res, n_atom)
                if atom.name in protein_backbone:
                    protein_neigbour_search.add_atom(atom, n_ch, n_res, n_atom)      

    return (
        neigbour_search,
        sugar_neigbour_search,
        phosphate_neigbour_search,
        base_neigbour_search,
        protein_neigbour_search
    )

def calculate_grad_and_angles(array: np.ndarray):

    x_dim, y_dim, z_dim = array.shape

    calculated_array = np.empty(array.shape, dtype=np.object_)

    for x in range(x_dim):
        for y in range(y_dim):
            for z in range(z_dim):
                
                neg_probe_x = x-1
                neg_probe_y = y-1
                neg_probe_z = z-1

                pos_probe_x = x+1
                pos_probe_y = y+1
                pos_probe_z = z+1

                if x == 0: 
                    neg_probe_x = x_dim-1
                    pos_probe_x = x + 1

                if x == x_dim - 1: 
                    pos_probe_x = 0
                    neg_probe_x = x - 1

                if y == 0: 
                    neg_probe_y = y_dim-1
                    pos_probe_y = y + 1

                if y == y_dim - 1: 
                    pos_probe_y = 0
                    neg_probe_y = y - 1

                if z == 0: 
                    neg_probe_z = z_dim-1
                    pos_probe_z = z + 1

                if z == z_dim - 1: 
                    pos_probe_z = 0
                    neg_probe_z = z - 1

                gradient_x = array[pos_probe_x][y][z] - array[neg_probe_x][y][z]
                gradient_y = array[x][pos_probe_y][z] - array[x][neg_probe_y][z]
                gradient_z = array[x][y][pos_probe_z] - array[x][y][neg_probe_z]

                magnitude = np.linalg.norm((gradient_x, gradient_y, gradient_z))
                theta = np.rad2deg(np.arccos((gradient_z/magnitude)))
                psi = np.rad2deg(np.arctan(gradient_y/gradient_x))

                theta = theta % 180
                psi = psi % 180

                data = { 
                    "magnitude": magnitude,
                    "theta": theta, 
                    "psi": psi
                }

                calculated_array[x][y][z] = data

    return calculated_array
                
def calculate_hog(array: np.array) -> Tuple[np.ndarray, np.ndarray]:
    assert array.shape == (8,8,8)

    array = calculate_grad_and_angles(array)

    x_dim, y_dim, z_dim = array.shape

    angle_step = 20
    number_of_bins = int(180 / angle_step)

    histogram_values = np.arange(0,180,20,dtype=np.int64)
    theta_histogram = np.column_stack((histogram_values, np.zeros(number_of_bins)))
    psi_histogram = np.column_stack((histogram_values, np.zeros(number_of_bins)))

    for x in range(x_dim): 
        for y in range(y_dim): 
            for z in range(z_dim):
                data = array[x][y][z]
                theta = data["theta"]
                psi = data["psi"]
                magnitude = data["magnitude"]

                theta_upper = int(theta / angle_step)
                theta_lower = int(theta // angle_step)


                psi_upper = int(psi / angle_step)
                psi_lower = int(psi // angle_step)

                if theta_upper > number_of_bins: 
                    theta_upper = 0

                if psi_upper > number_of_bins: 
                    psi_upper = 0

                theta_lower_bin = theta_histogram[theta_lower][0]
                psi_lower_bin = psi_histogram[psi_lower][0]

                theta_lower_delta = theta - theta_lower_bin
                psi_lower_delta = psi - psi_lower_bin

                theta_lower_prop = theta_lower_delta / 20
                theta_upper_prop = 1 - theta_lower_prop

                psi_lower_prop = psi_lower_delta / 20
                psi_upper_prop = 1 - psi_lower_prop

                theta_histogram[theta_lower][1] += magnitude * theta_lower_prop
                theta_histogram[theta_upper][1] += magnitude * theta_upper_prop

                psi_histogram[psi_lower][1] += magnitude * psi_lower_prop
                psi_histogram[psi_upper][1] += magnitude * psi_upper_prop

    return theta_histogram, psi_histogram

def find_atomic_positions(grid: gemmi.FloatGrid, transform: gemmi.Transform, pdb_code: str): 

    array = grid.array

    try:
        structure = data.import_pdb(pdb_code)
    except:
        print("[FAILED]:", pdb_code)
        return
    
    (
        neigbour_search,
        sugar_neigbour_search,
        phosphate_neigbour_search,
        base_neigbour_search,
        protein_neigbour_search
    ) = _initialise_neighbour_search(structure)

    radius = 1.5

    output = []

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for k in range(array.shape[2]):
                index_pos = gemmi.Vec3(i, j, k)
                position = gemmi.Position(transform.apply(index_pos))

                any_bases = base_neigbour_search.find_atoms(position, "\0", radius=radius)
                any_sugars = sugar_neigbour_search.find_atoms(position, "\0", radius=radius)
                any_phosphate = phosphate_neigbour_search.find_atoms(
                    position, "\0", radius=1
                )
                any_protein = protein_neigbour_search.find_atoms(position, "\0", radius=radius)

                base_mask = 1.0 if len(any_bases) > 1 else 0.0
                sugar_mask = 1.0 if len(any_sugars) > 1 else 0.0
                phosphate_mask = 1.0 if len(any_phosphate) > 1 else 0.0
                protein_mask = 1.0 if len(any_protein) > 1 else 0.0

                sum_ = np.sum((base_mask, sugar_mask, phosphate_mask, protein_mask))

                if sum_ == 1:
                    density_array: np.ndarray = np.array(grid.get_subarray(
                        start=[i-4, j-4, k-4], shape=[8, 8, 8]
                    ))

                    theta_hist, psi_hist = calculate_hog(density_array)
                    
                    concat_hist = np.concatenate((theta_hist[:,1], psi_hist[:,1]))

                    if base_mask == 1.0:
                        output.append([concat_hist, "base"])

                    if sugar_mask == 1.0: 
                        output.append([concat_hist, "sugar"])

                    if phosphate_mask == 1.0:
                        output.append([concat_hist, "phosphate"])

                    if protein_mask == 1.0: 
                        output.append([concat_hist, "protein"])

    return output

def write_output(output, pdb_code):
    output_dir = "hog_dataset"
    output_path = os.path.join(output_dir, f"{pdb_code}.csv")
    with open(output_path, 'w') as output_file: 
        header = "theta_0,theta_20,theta_40,theta_60,theta_80,theta_100,theta_120,theta_140,theta_160,psi_0,psi_20,psi_40,psi_60,psi_80,psi_100,psi_120,psi_140,psi_160,classification\n"
        output_file.write(header)
        for entry in output:
            histograms, classification = entry
            for value in histograms:
                output_file.write(f"{value},")
            output_file.write(f"{classification}\n")

def worker(data):
    map_path, pdb_code = data
    interpolated_map, transform = utils.load_and_interpolate_map(map_path)
    output = find_atomic_positions(interpolated_map, transform, pdb_code)

    if output: 
        write_output(output,pdb_code)

def main(): 
    map_dir = "data/DNA_test_structures/external_test_maps/map"
    map_list = [(os.path.join(map_dir,x), x.replace(".map","")) for x in os.listdir(map_dir) if '.map' in x]

    with Pool(64) as pool:
        x = list(tqdm(pool.imap_unordered(worker, map_list), total=len(map_list)))
        

if __name__ == "__main__":
    now = time.time()
    main()
    end = time.time()
    print("Time taken", end-now)
    