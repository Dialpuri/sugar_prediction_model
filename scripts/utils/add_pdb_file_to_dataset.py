import os
import shutil

def main(): 
    dataset_path = "low_res_dataset"

    for path in os.scandir(dataset_path):
        pdb_code = path.name
        pdb_file_path = "/vault/pdb/pdb" + pdb_code + ".ent" 

        shutil.copy(pdb_file_path, path.path)


if __name__ == "__main__":
    main()