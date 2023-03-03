from dataclasses import dataclass

@dataclass
class Parameters:
    pdb_location: str = "./data/DNA_test_structures/PDB_Files"
    mtz_location: str = "./data/DNA_test_structures/MTZ_Files"
    pdb_file_ending: str = "pdb"
    mtz_file_ending: str = "mtz"
    map_out_dir: str = "./data/DNA_test_structures/maps"
    pdb_list_file_path: str = "./data/DNA_test_structures/file_list_all.txt"
    ns_radius: int = 2
    maps_dir: str = "./data/DNA_test_structures/maps"
