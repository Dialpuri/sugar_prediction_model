from typing import Tuple
import time
import gemmi 
import numpy as np 
import utils
import math 
import timeit
import seaborn as sns
import matplotlib.pyplot as plt


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
                
def array_to_chunks(array: np.array) -> Tuple[np.ndarray, np.ndarray]:

    x, y, z = array.shape

    block_size = 8

    resized_x = int(x/block_size)
    resized_y = int(y/block_size)
    resized_z = int(z/block_size)

    resized_array = np.resize(array, (block_size*resized_x, block_size*resized_y, block_size*resized_z))


def calculate_hog(array: np.array):
    assert array.shape == (8,8,8)

    x_dim, y_dim, z_dim = array.shape

    angle_step = 20
    number_of_bins = int(180 / angle_step)

    histogram_values = np.arange(0,180,20)
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


def main(): 
    file_path = "data/test_maps/1hr2.map"
    interpolated_map: gemmi.FloatGrid = utils.load_and_interpolate_map(file_path)

    calculated_array = calculate_grad_and_angles(interpolated_map.array[40:48,12:20,:8])

    test_chunk = calculated_array
    theta_hist, psi_hist = calculate_hog(test_chunk)
    
    # sns.jointplot(x=theta_hist[:,1], y=psi_hist[:,1], kind="hist", space=0, bins=(9,9))
    # plt.show()


if __name__ == "__main__":
    now = time.time()
    main()
    end = time.time()
    print("Time taken", end-now)
    