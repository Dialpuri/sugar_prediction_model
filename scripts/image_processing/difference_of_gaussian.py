import gemmi 
import numpy as np
import sys
sys.path.append('/home/jordan/dev/sugar_position_pred/scripts/interpolated_model')
import utils
import math



def gaussian_3d_at_ijk(i: float, j: float, k: float, sigma: float) -> float:
    kernel = (1 / (sigma * 2 * math.sqrt(2 * math.pi))) * (math.exp((-((pow(i,2)+(pow(j,2))+(pow(k,2)))))/(2*(pow(sigma,2)))));
    return kernel

def generate_gaussian_kernel(sigma: float, matrix_size: float) -> np.ndarray:

    kernel: np.ndarray = np.zeros((matrix_size, matrix_size, matrix_size), dtype=np.float32)

    matrix_dimension: int = matrix_size // 2

    for index_i, value_i in enumerate(range(-matrix_dimension, matrix_dimension+1, 1)):
        for index_j, value_j in enumerate(range(-matrix_dimension, matrix_dimension+1, 1)):
            for index_k, value_k in enumerate(range(-matrix_dimension, matrix_dimension+1, 1)):
                kernel[index_i][index_j][index_k] = gaussian_3d_at_ijk(value_i, value_j, value_k, sigma)
    
    return kernel


def gaussian_blur(array: np.ndarray, sigma: float, kernel_size: float = 3) -> np.ndarray:
    kernel = generate_gaussian_kernel(sigma, kernel_size)

    fft_kernel = np.fft.fftn(kernel, s=array.shape)
    fft_array = np.fft.fftn(array)

    convolution = np.fft.ifftn(fft_kernel*fft_array).real

    return convolution.astype(np.float32)


def load_and_interpolate_map(file_path: str, grid_spacing=0.7) -> gemmi.FloatGrid:
    map : gemmi.Ccp4Map = gemmi.read_ccp4_map(file_path)
    grid = map.grid
    box: gemmi.PositionBox = utils.get_bounding_box(grid)
    size: gemmi.Position = box.get_size()

    num_x = int(size.x / grid_spacing)
    num_y = int(size.y / grid_spacing)
    num_z = int(size.z / grid_spacing)

    array = np.zeros((num_x, num_y, num_z), dtype=np.float32)
    scale = gemmi.Mat33(
        [[grid_spacing, 0, 0], [0, grid_spacing, 0], [0, 0, grid_spacing]]
    )

    transform: gemmi.Transform = gemmi.Transform(scale, box.minimum)
    grid.interpolate_values(array, transform)
    cell: gemmi.UnitCell = gemmi.UnitCell(size.x, size.y, size.z, 90, 90, 90)
    interpolated_grid: gemmi.FloatGrid = gemmi.FloatGrid(array, cell)

    return interpolated_grid

def save_map(output_path: str, array: np.ndarray, unit_cell: gemmi.UnitCell): 
    ccp4 = gemmi.Ccp4Map()
    grid = gemmi.FloatGrid(array, unit_cell, gemmi.SpaceGroup("P1"))
    ccp4.grid = grid
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map(output_path)


def main(): 
    file_path = "data/DNA_test_structures/external_test_maps/map/1hr2.map"
    interpolated_map: gemmi.FloatGrid = load_and_interpolate_map(file_path)

    save_map("data/blurred_maps/1hr2_noblur.map", interpolated_map.array, interpolated_map.unit_cell)

    blur_1 = gaussian_blur(interpolated_map.array, 2)
    blur_2 = gaussian_blur(interpolated_map.array, 3)

    difference = np.subtract(blur_1, blur_2)

    save_map("data/blurred_maps/1hr2_dog_2_3.map", difference, interpolated_map.unit_cell)



if __name__ == "__main__":
    main()
