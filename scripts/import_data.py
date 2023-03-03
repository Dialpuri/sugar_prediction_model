import random
import math
import gemmi
import os
from dataclasses import dataclass
from multiprocessing import Pool
from tqdm import tqdm
import parameters


def import_pdb(pdb_code: str) -> gemmi.Structure:
    params = parameters.Parameters()
    pdb_file_path = os.path.join(params.pdb_location, f"{pdb_code}.{params.pdb_file_ending}")
    try:
        structure = gemmi.read_structure(pdb_file_path)
    except (RuntimeError, ValueError) as e:
        print(f"{pdb_code} raised {e}")
        return
    return structure


def import_map_from_mtz(pdb_code: str) -> gemmi.FloatGrid:
    mtz_file_path = os.path.join(params.mtz_location, f"{pdb_code}_phases.{params.mtz_file_ending}")
    try:
        mtz = gemmi.read_mtz_file(mtz_file_path)
    except (RuntimeError, ValueError) as e:
        print(f"{pdb_code} raised {e}")
        return
    return mtz.transform_f_phi_to_map("FWT", "PHWT")


def read_pdb_list() -> list[str]:
    pdb_list = []
    with open(params.pdb_list_file_path) as pdb_list_file:
        for line in pdb_list_file:
            pdb_list.append(line.strip())
    return pdb_list


def write_map(input_: tuple[str, str]):
    pdb_code = input_[0]
    assignment = input_[1]
    map_grid = import_map_from_mtz(pdb_code)
    output_path = os.path.join(params.map_out_dir, f"{pdb_code}_{assignment}.map")

    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = map_grid
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map(output_path)


def main():
    pdb_list = read_pdb_list()

    random.shuffle(pdb_list)

    TEST_TRAIN_SPLIT = 0.8

    train_index = math.floor(len(pdb_list) * TEST_TRAIN_SPLIT)

    train_list = [(x, "train") for x in pdb_list[:train_index]]
    test_list = [(x, "test") for x in pdb_list[train_index:]]


    with Pool() as pool:
        list(tqdm(pool.imap_unordered(write_map, train_list), total=len(train_list)))

    with Pool() as pool:
        list(tqdm(pool.imap_unordered(write_map, test_list), total=len(test_list)))



if __name__ == "__main__":
    params = parameters.Parameters()
    main()
