from dataclasses import dataclass

@dataclass
class Parameters:
    # pdb_location: str = "./data/DNA_test_structures/PDB_Files"
    pdb_location: str = "/vault/pdb/"
    # pdb_location: str = "data/pdb/"
    mtz_location: str = "./data/DNA_test_structures/MTZ_Files"
    pdb_file_ending: str = "ent"
    pdb_prefix: str = "pdb"
    mtz_file_ending: str = "mtz"
    map_out_dir: str = "./data/DNA_test_structures/maps"
    pdb_list_file_path: str = "./data/DNA_test_structures/file_list_all.txt"
    ns_radius: int = 2
    maps_dir: str = "./data/DNA_test_structures/maps"
    precompute_list_dir: str = "./data/DNA_test_structures/16x16x16_box_list.csv"
