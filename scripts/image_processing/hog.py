from multiprocessing import Pool
import time
import gemmi
import numpy as np
from tqdm import tqdm
from scipy.ndimage import convolve1d
import os
import sys

sys.path.append("/home/jordan/dev/sugar_position_pred/scripts/interpolated_model")
import utils
import import_data as data


def _initialise_neighbour_search(structure: gemmi.Structure, radius: int = 3):
    neigbour_search = gemmi.NeighborSearch(structure[0], structure.cell, 1).populate()

    sugar_neigbour_search = gemmi.NeighborSearch(structure[0], structure.cell, radius)
    phosphate_neigbour_search = gemmi.NeighborSearch(
        structure[0], structure.cell, radius
    )
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
    protein_backbone = [
        "CA",
        "CB",
        "C",
        "OXT",
    ]

    all_sugar_neigbour_search = gemmi.NeighborSearch(
        structure[0], structure.cell, radius
    )
    # all_sugar_atoms = sugar_atoms + phosphate_atoms + base_atoms
    all_sugar_atoms = phosphate_atoms

    for n_ch, chain in enumerate(structure[0]):
        for n_res, res in enumerate(chain):
            for n_atom, atom in enumerate(res):
                if atom.name in protein_backbone:
                    protein_neigbour_search.add_atom(atom, n_ch, n_res, n_atom)
                if atom.name in all_sugar_atoms:
                    all_sugar_neigbour_search.add_atom(atom, n_ch, n_res, n_atom)

    return (protein_neigbour_search, all_sugar_neigbour_search)


def calculate_histograms(array: np.ndarray, box_size: int = 8):
    angle_step = 20
    number_of_bins = int(180 / angle_step)

    histogram_values = np.arange(0, 180, 20)
    theta_histogram = np.column_stack((histogram_values, np.zeros(number_of_bins)))
    psi_histogram = np.column_stack((histogram_values, np.zeros(number_of_bins)))

    x_vector = convolve1d(array, np.array([-1, 0, 1]), axis=0, mode="nearest")
    y_vector = convolve1d(array, np.array([-1, 0, 1]), axis=1, mode="nearest")
    z_vector = convolve1d(array, np.array([-1, 0, 1]), axis=2, mode="nearest")

    magnitudes = np.zeros((box_size, box_size, box_size))
    for i in range(box_size):
        for j in range(box_size):
            for k in range(box_size):
                magnitudes[i, j, k] = np.linalg.norm((x_vector[i,j,k], y_vector[i,j,k],z_vector[i,j,k]))
               
    theta = np.zeros((box_size, box_size, box_size))
    phi = np.zeros((box_size, box_size, box_size))

    for i in range(box_size):
        for j in range(box_size):
            for k in range(box_size):
                theta[i, j, k] = (
                    np.rad2deg(np.arccos(z_vector[i, j, k] / magnitudes[i, j, k])) % 180
                )
                phi[i, j, k] = (
                    np.rad2deg(np.arctan(y_vector[i, j, k] / x_vector[i, j, k])) % 180
                )

    

    for m, t, p in zip(magnitudes.flatten(), theta.flatten(), phi.flatten()):

        if np.isnan(np.sum(t)) or np.isnan(np.sum(p)):
            continue

        theta_upper = int((t // angle_step) + 1)
        theta_lower = int(t // angle_step)

        psi_upper = int((p / angle_step) + 1)
        psi_lower = int(p // angle_step)

        if theta_upper >= number_of_bins:
            theta_upper = 0

        if psi_upper >= number_of_bins:
            psi_upper = 0

        theta_lower_bin = theta_histogram[theta_lower][0]
        psi_lower_bin = psi_histogram[psi_lower][0]

        theta_lower_delta = t - theta_lower_bin
        psi_lower_delta = p - psi_lower_bin

        theta_lower_prop = theta_lower_delta / 20
        theta_upper_prop = 1 - theta_lower_prop

        psi_lower_prop = psi_lower_delta / 20
        psi_upper_prop = 1 - psi_lower_prop

        # print(theta_lower_prop, theta_upper_prop, m, abs(m))
        # print(p, psi_lower_bin, psi_lower_delta, psi_lower_prop, psi_upper_prop, m, abs(m))

        theta_histogram[theta_lower][1] += m * theta_lower_prop
        theta_histogram[theta_upper][1] += m * theta_upper_prop

        psi_histogram[psi_lower][1] += m * psi_lower_prop
        psi_histogram[psi_upper][1] += m * psi_upper_prop

    return theta_histogram, psi_histogram

def traverse_atoms(pdb_code: str, interpolated_map: gemmi.FloatGrid, transform: gemmi.Transform, box_size: int=8): 

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
    protein_backbone = [
        "CA",
        "CB",
        "C",
        "OXT",
    ]

    combined_sugar_atoms = sugar_atoms + phosphate_atoms + base_atoms

    output = []

    try:
        structure = data.import_pdb(pdb_code)
    except:
        print("[FAILED]:", pdb_code)
        return

    for n_ch, chain in enumerate(structure[0]):
        for n_res, res in enumerate(chain):
            for n_atom, atom in enumerate(res):
                if atom.name in protein_backbone or atom.name in combined_sugar_atoms:
                    raw_pos = atom.pos
                    int_pos = gemmi.Position(transform.apply(raw_pos))

                    nearest_point = interpolated_map.get_nearest_point(int_pos)

                    i_pos = int(nearest_point.u - (box_size / 2))
                    j_pos = int(nearest_point.v - (box_size / 2))
                    k_pos = int(nearest_point.w - (box_size / 2))

                    density_array: np.ndarray = np.array(
                        interpolated_map.get_subarray(
                            start=[i_pos, j_pos, k_pos],
                            shape=[box_size, box_size, box_size],
                        )
                    )

                    theta_hist, psi_hist = calculate_histograms(density_array, box_size)

                    concat_hist = np.concatenate((theta_hist[:, 1], psi_hist[:, 1]))

                    if atom.name in protein_backbone:
                        output.append([concat_hist, "protein"])
                    if atom.name in combined_sugar_atoms: 
                        output.append([concat_hist, "sugar"])

    return output


def find_atomic_positions(
    grid: gemmi.FloatGrid,
    transform: gemmi.Transform,
    pdb_code: str,
    radius: int = 2,
    box_size: int = 25,
):
    array = grid.array

    try:
        structure = data.import_pdb(pdb_code)
    except:
        print("[FAILED]:", pdb_code)
        return

    (
        protein_neigbour_search,
        all_sugar_neighbour_search,
    ) = _initialise_neighbour_search(structure)

    array = np.clip(array, 0, None)

    output = []

    for i in range(0, array.shape[0], 4):
        for j in range(0, array.shape[1], 4):
            for k in range(0, array.shape[2], 4):
                index_pos = gemmi.Vec3(i, j, k)
                position = gemmi.Position(transform.apply(index_pos))

                any_protein = protein_neigbour_search.find_atoms(
                    position, "\0", radius=radius
                )
                any_sugars = all_sugar_neighbour_search.find_atoms(
                    position, "\0", radius=radius
                )

                protein_mask = 1.0 if len(any_protein) > 1 else 0.0
                sugar_mask = 1.0 if len(any_sugars) > 1 else 0.0

                sum_ = protein_mask + sugar_mask

                if sum_ == 1:
                    i_pos = int(i - (box_size / 2))
                    j_pos = int(j - (box_size / 2))
                    k_pos = int(k - (box_size / 2))
                    density_array: np.ndarray = np.array(
                        grid.get_subarray(
                            start=[i_pos, j_pos, k_pos],
                            shape=[box_size, box_size, box_size],
                        )
                    )

                    theta_hist, psi_hist = calculate_histograms(density_array, box_size)

                    concat_hist = np.concatenate((theta_hist[:, 1], psi_hist[:, 1]))

                    if protein_mask:
                        output.append([concat_hist, "p"])

                    if sugar_mask:
                        output.append([concat_hist, "s"])

    return output


def write_output(output, pdb_code):
    output_dir = "hog_dataset"
    output_path = os.path.join(output_dir, f"{pdb_code}.csv")
    with open(output_path, "w") as output_file:
        header = "theta_0,theta_20,theta_40,theta_60,theta_80,theta_100,theta_120,theta_140,theta_160,psi_0,psi_20,psi_40,psi_60,psi_80,psi_100,psi_120,psi_140,psi_160,classification\n"
        output_file.write(header)
        for entry in output:
            histograms, classification = entry
            for value in histograms:
                output_file.write(f"{value},")
            output_file.write(f"{classification}\n")


def worker_pointwise(data):
    map_path, pdb_code = data
    interpolated_map, transform = utils.load_and_interpolate_map(
        map_path, grid_spacing=0.2
    )
    output = find_atomic_positions(
        interpolated_map, transform, pdb_code, radius=2, box_size=25
    )

    if output:
        write_output(output, pdb_code)


def generate_hog_list_pointwise():
    map_dir = "data/DNA_test_structures/external_test_maps/map"
    map_list = [
        (os.path.join(map_dir, x), x.replace(".map", ""))
        for x in os.listdir(map_dir)
        if ".map" in x
    ]

    with Pool() as pool:
        x = list(tqdm(pool.imap_unordered(worker_pointwise, map_list), total=len(map_list)))

def worker_atomwise(data):
    map_path, pdb_code = data
    interpolated_map, transform = utils.load_and_interpolate_map(
        map_path, grid_spacing=0.5
    )
    output = traverse_atoms(pdb_code=pdb_code, interpolated_map=interpolated_map, transform=transform, box_size=10)

    if output:
        write_output(output, pdb_code)


def generate_hog_list_atomwise():
    map_dir = "data/DNA_test_structures/external_test_maps/map"
    map_list = [
        (os.path.join(map_dir, x), x.replace(".map", ""))
        for x in os.listdir(map_dir)
        if ".map" in x
    ]

    with Pool() as pool:
        x = list(tqdm(pool.imap_unordered(worker_atomwise, map_list), total=len(map_list)))


    # worker(map_list[0])

def x(): 
    structure = data.import_pdb("1fxl")
    interpolated_map, transform = utils.load_and_interpolate_map(
            "data/DNA_test_structures/external_test_maps/map/1fxl.map", grid_spacing=0.2
        )
    
    protein_histograms, sugar_histograms = traverse_atoms(structure=structure, interpolated_map=interpolated_map, transform=transform)
    p = np.mean(protein_histograms, axis=0)
    s = np.mean(sugar_histograms, axis=0)

    py = np.outer(p[:9], p[9:])
    ps = np.outer(s[:9], s[9:])
    import matplotlib.pyplot as plt
    fig, (ax0, ax1) = plt.subplots(2)

    c0 = ax0.matshow(py)
    c1 = ax1.matshow(ps)
    fig.colorbar(c0)
    fig.colorbar(c1)
    plt.savefig("data/hog/graphs/proteins.png")


    
    


def main():
    # generate_hog_list()
    # x()
    generate_hog_list_atomwise()

if __name__ == "__main__":
    now = time.time()
    main()
    end = time.time()
    print("Time taken", end - now)
