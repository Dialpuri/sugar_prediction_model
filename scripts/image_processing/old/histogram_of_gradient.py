import csv
from multiprocessing import Pool
import os
from typing import Tuple
import time
import gemmi 
import sys
import pandas as pd
from tqdm import tqdm
sys.path.append("/home/jordan/dev/sugar_position_pred/scripts/interpolated_model")
import utils
import import_data as data
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.model_selection import train_test_split
from tqdm import tqdm 
import seaborn as sns

def _initialise_neighbour_search(structure: gemmi.Structure, radius: int = 3):
    neigbour_search = gemmi.NeighborSearch(structure[0], structure.cell, 1).populate()

    sugar_neigbour_search = gemmi.NeighborSearch(structure[0], structure.cell, radius)
    phosphate_neigbour_search = gemmi.NeighborSearch(structure[0], structure.cell, radius)
    base_neigbour_search = gemmi.NeighborSearch(structure[0], structure.cell, radius)
    protein_neigbour_search = gemmi.NeighborSearch(structure[0], structure.cell, radius)

    all_sugar_neigbour_search = gemmi.NeighborSearch(structure[0], structure.cell, radius)

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
    protein_backbone = ["CA", "CB", "OXT", "O", "C", "N"]

    all_sugar_atoms = sugar_atoms + phosphate_atoms + base_atoms

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
                if atom.name in all_sugar_atoms:
                    all_sugar_neigbour_search.add_atom(atom, n_ch, n_res, n_atom)      

    return (
        neigbour_search,
        sugar_neigbour_search,
        phosphate_neigbour_search,
        base_neigbour_search,
        protein_neigbour_search,
        all_sugar_neigbour_search
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
    # assert array.shape == (8,8,8)

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

                if np.isnan(psi) or np.isnan(theta):
                    continue

                theta_upper = int((theta // angle_step) + 1)
                theta_lower = int(theta // angle_step)

                psi_upper = int((psi / angle_step) + 1)
                psi_lower = int(psi // angle_step)

                if theta_upper >= number_of_bins: 
                    theta_upper = 0

                if psi_upper >= number_of_bins: 
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

    box_size = 12

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
        protein_neigbour_search,
        all_sugar_neigbour_search
    ) = _initialise_neighbour_search(structure)

    radius = 1

    output = []

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for k in range(array.shape[2]):
                index_pos = gemmi.Vec3(i, j, k)
                position = gemmi.Position(transform.apply(index_pos))

                any_sugars = all_sugar_neigbour_search.find_atoms(position, "\0", radius=radius)

                # any_bases = base_neigbour_search.find_atoms(position, "\0", radius=radius)
                # any_sugars = sugar_neigbour_search.find_atoms(position, "\0", radius=radius)
                # any_phosphate = phosphate_neigbour_search.find_atoms(
                #     position, "\0", radius=1
                # )
                any_protein = protein_neigbour_search.find_atoms(position, "\0", radius=radius)

                # base_mask = 1.0 if len(any_bases) > 1 else 0.0
                # sugar_mask = 1.0 if len(any_sugars) > 1 else 0.0
                # phosphate_mask = 1.0 if len(any_phosphate) > 1 else 0.0
                protein_mask = 1.0 if len(any_protein) > 1 else 0.0
                sugar_mask = 1.0 if len(any_sugars) > 1 else 0.0

                if protein_mask or sugar_mask:
                    i_pos = int(i-(box_size/2))
                    j_pos = int(j-(box_size/2))
                    k_pos = int(k-(box_size/2))
                    density_array: np.ndarray = np.array(grid.get_subarray(
                        start=[i_pos, j_pos, k_pos], shape=[box_size, box_size, box_size]
                    ))
                    theta_hist, psi_hist = calculate_hog(density_array)
                    
                    concat_hist = np.concatenate((theta_hist[:,1], psi_hist[:,1]))

                    if protein_mask:
                        output.append([concat_hist, "protein"])
                    
                    if sugar_mask:
                        output.append([concat_hist, "sugar"])


                # sum_ = np.sum((base_mask, sugar_mask, phosphate_mask, protein_mask))

                # if sum_ == 1:
                #     density_array: np.ndarray = np.array(grid.get_subarray(
                #         start=[i-(box_size/2), j-(box_size/2), k-(box_size/2)], shape=[box_size, box_size, box_size]
                #     ))

                #     theta_hist, psi_hist = calculate_hog(density_array)
                    
                #     concat_hist = np.concatenate((theta_hist[:,1], psi_hist[:,1]))

                #     if base_mask == 1.0:
                #         output.append([concat_hist, "base"])

                #     if sugar_mask == 1.0: 
                #         output.append([concat_hist, "sugar"])

                #     if phosphate_mask == 1.0:
                #         output.append([concat_hist, "phosphate"])

                #     if protein_mask == 1.0: 
                #         output.append([concat_hist, "protein"])

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

def generate_hog_list(): 
    map_dir = "data/DNA_test_structures/external_test_maps/map"
    map_list = [(os.path.join(map_dir,x), x.replace(".map","")) for x in os.listdir(map_dir) if '.map' in x]

    with Pool() as pool:
        x = list(tqdm(pool.imap_unordered(worker, map_list), total=len(map_list)))
        
def generate_test_train_split(): 
    output_df = pd.DataFrame(columns=[
        "theta_0","theta_20","theta_40","theta_60","theta_80","theta_100","theta_120","theta_140","theta_160","psi_0","psi_20","psi_40","psi_60","psi_80","psi_100","psi_120","psi_140","psi_160","classification"

    ])
    for path in tqdm(os.scandir("hog_dataset"), total=len(os.listdir("hog_dataset"))):
        df = pd.read_csv(path.path)
        output_df =  pd.concat([output_df, df], axis=0, ignore_index=True)
    
    train, test = train_test_split(output_df, shuffle=True, test_size=0.2)

    print(f"Train length is {len(train)}")
    print(f"Test length is {len(test)}")

    train.to_csv("data/hog/train_hog.csv", index=False)
    test.to_csv("data/hog/test_hog.csv", index=False)

def check_data():
    df = pd.read_csv("data/hog/train_hog.csv")

    grouped_df = df.groupby("classification")

    fig, axs = plt.subplots(2)

    for (index, data), ax in zip(grouped_df, axs):
        means = data[df.columns[:-1]].mean().to_list()

        mean_2d = np.outer(means[:9], means[9:])
        c = ax.matshow(mean_2d)
        fig.colorbar(c)
        
    plt.savefig(f"data/hog/graphs/meanp+s.png")


def generate_2d_hist():

    array_list = []

    with open("data/hog/train_hog.csv") as input_file:
        r = csv.reader(input_file)
        for index, row in enumerate(r):
            if index == 0: continue

            theta = row[:9]
            psi = row[9:-1]

            classification = row[-1]
            if classification != "phosphate":
                continue

            array = np.zeros((9,9), dtype=np.float32)

            for t in range(len(theta)):
                for p in range(len(psi)):
                    array[t][p] = float(theta[t]) * float(psi[p])

            array_list.append(array)

            if index > 100: break

    means = np.mean(array_list, axis=0)
    plt.matshow(means)
    plt.savefig("data/hog/graphs/test_phos.png")


if __name__ == "__main__":
    now = time.time()

    # generate_hog_list()
    generate_test_train_split()
    check_data()
    # generate_2d_hist()

    end = time.time()
    print("Time taken", end-now)
    