import logging
import os
from typing import List, Tuple
import tensorflow as tf
import tensorflow_addons as tfa
import utils
import import_data as data
import numpy as np
import gemmi
from tqdm import tqdm
from dataclasses import dataclass

@dataclass
class Scores:
    positive: int = 0
    negative: int = 0
    false_positive: int = 0
    false_negative: int = 0

class TestModel:
    cache_path = "cache"
    predicted_name = "predicted.npy"
    sugar_name = "sugar.npy"
    base_name = "base.npy"
    phos_name = "phosphate.npy"


    def __init__(self, model_dir: str, use_cache: bool = True):
        self.use_cache: bool = use_cache
        self.model_dir: str = model_dir
        self.model_name: str = model_dir.split("/")[-1]

        self.base_score: Scores = Scores()

        self.predicted_map: np.ndarray = None
        self.sugar_map: np.ndarray = None
        self.phosphate_map: np.ndarray = None
        self.base_map: np.ndarray = None

        self.pdb_code = ""

        self.na: float = 0
        self.nb: float = 0
        self.nc: float = 0
        self.translation_list: List[List[int, int, int]] = []

        self.model: tf.keras.Model = None

        self.interpolated_grid: gemmi.FloatGrid = None
        self.raw_grid: gemmi.FloatGrid = None
        self.map: gemmi.Ccp4Map = None
        self.structure: gemmi.Structure = None
        self.transform: gemmi.Transform = None

    def __repr__(self):
        return f"Test Model({self.pdb_code}, s: {self.sugar_score}, {self.sugar_false_score} - p: {self.phosphate_score}, {self.phosphate_false_score}"

    def __str__(self):
        return f"""Test Model of {self.pdb_code} """

    def make_prediction(self, map_path: str, pdb_code: str, use_raw_values: bool = False):
        self.pdb_code = pdb_code
        cache_path = os.path.join(self.cache_path, pdb_code, self.model_name)

        if not self.use_cache or not os.path.isdir(cache_path):
            logging.info("No cache found")
            self._load_model()
            self._load_map(map_path=map_path)
            self._load_structure(pdb_code=pdb_code)
            self._interpolate_grid()
            self._calculate_translations(overlap=16)
            self._predict(raw_values=use_raw_values)

            self._generate_sugar_map()
            self._generate_phosphate_map()
            self._generate_base_map()

            self.base_score = self._score_predicted_map()
            self._save_cache(cache_path)
        else:
            self._load_cache(cache_path)
            self.base_score = self._score_predicted_map()

    def save_score(self, output_dir: str):
        model_name = self.model_dir.split("/")[-1]
        output_dir = os.path.join(output_dir, model_name)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        output_path = os.path.join(output_dir, f"{self.pdb_code}.csv")
        with open(output_path, "w") as output_file:
            output_file.write("PDB,Positive,Negative,FalsePositive,FalseNegative\n")
            output_file.write(
                f"{self.pdb_code},{self.base_score.positive:.3f},{self.base_score.negative:.3f},{self.base_score.false_positive:.3f},{self.base_score.false_negative:.3f}"
            )

    def save_predicted_map(
        self, output_dir: str, grid_spacing: float = 0.7, original: bool = False
    ):
        size_x = self.predicted_map.shape[0] * grid_spacing
        size_y = self.predicted_map.shape[1] * grid_spacing
        size_z = self.predicted_map.shape[2] * grid_spacing

        array_cell = gemmi.UnitCell(size_x, size_y, size_z, 90, 90, 90)
        array_grid = gemmi.FloatGrid(self.predicted_map, array_cell)

        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = array_grid
        ccp4.update_ccp4_header()

        output_path = os.path.join(output_dir, f"{self.pdb_code}_predicted.map")
        ccp4.write_ccp4_map(output_path)

    def save_sugar_map(
        self, output_dir: str, grid_spacing: float = 0.7, original: bool = False
    ):
        size_x = self.sugar_map.shape[0] * grid_spacing
        size_y = self.sugar_map.shape[1] * grid_spacing
        size_z = self.sugar_map.shape[2] * grid_spacing

        logging.debug(f"Saving sugar map size x,y,z : {size_x}, {size_y}, {size_z}")

        array_cell = gemmi.UnitCell(size_x, size_y, size_z, 90, 90, 90)
        array_grid = gemmi.FloatGrid(self.sugar_map, array_cell)

        logging.debug(f"Array grid unit cell : {array_grid.unit_cell}")

        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = array_grid
        ccp4.update_ccp4_header()

        output_path = os.path.join(output_dir, f"{self.pdb_code}_sugars.map")
        ccp4.write_ccp4_map(output_path)

    def save_base_map(
        self, output_dir: str, grid_spacing: float = 0.7, original: bool = False
    ):
        size_x = self.base_map.shape[0] * grid_spacing
        size_y = self.base_map.shape[1] * grid_spacing
        size_z = self.base_map.shape[2] * grid_spacing

        logging.debug(f"Saving base map size x,y,z : {size_x}, {size_y}, {size_z}")

        array_cell = gemmi.UnitCell(size_x, size_y, size_z, 90, 90, 90)
        array_grid = gemmi.FloatGrid(self.base_map, array_cell)

        logging.debug(f"Array grid unit cell : {array_grid.unit_cell}")

        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = array_grid
        ccp4.update_ccp4_header()

        output_path = os.path.join(output_dir, f"{self.pdb_code}_base.map")
        ccp4.write_ccp4_map(output_path)

    def save_phosphate_map(
        self, output_dir: str, grid_spacing: float = 0.7, original: bool = False
    ):
        size_x = self.phosphate_map.shape[0] * grid_spacing
        size_y = self.phosphate_map.shape[1] * grid_spacing
        size_z = self.phosphate_map.shape[2] * grid_spacing

        logging.debug(f"Saving phosphate map size x,y,z : {size_x}, {size_y}, {size_z}")

        array_cell = gemmi.UnitCell(size_x, size_y, size_z, 90, 90, 90)
        array_grid = gemmi.FloatGrid(self.phosphate_map, array_cell)

        logging.debug(f"Array grid unit cell : {array_grid.unit_cell}")

        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = array_grid
        ccp4.update_ccp4_header()

        output_path = os.path.join(output_dir, f"{self.pdb_code}_phosphate.map")
        ccp4.write_ccp4_map(output_path)

    def save_interpolated_map(self, output_dir: str, grid_spacing: float = 0.7):
        logging.info("Saving interpolated map")
        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = self.interpolated_grid
        ccp4.update_ccp4_header()

        output_path = os.path.join(output_dir, f"{self.pdb_code}_interpolated.map")
        ccp4.write_ccp4_map(output_path)

    def save_maps(self, output_dir: str, original=True):
        output_dir = os.path.join(output_dir, self.pdb_code, self.model_name)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        self.save_predicted_map(output_dir=output_dir, original=original)
        self.save_sugar_map(output_dir=output_dir, original=original)
        self.save_phosphate_map(output_dir=output_dir, original=original)
        self.save_base_map(output_dir=output_dir, original=original)
        self.save_interpolated_map(output_dir=output_dir)

    def _load_model(self):

        logging.info(f"Loading model from dir: {self.model_dir}")

        if os.path.isdir(self.model_dir):
            logging.info("Loading model from model folder")
            print(self.model_dir, type(self.model_dir))
            self.model = tf.keras.models.load_model(
                self.model_dir,
                custom_objects={
                    "sigmoid_focal_crossentropy": tfa.losses.sigmoid_focal_crossentropy
                }
            )
        elif os.path.isfile(self.model_dir):
            logging.info("Loading model from weight file")
            self.model = tf.keras.models.load_model(
                self.model_dir,
                custom_objects={
                    "sigmoid_focal_crossentropy": tfa.losses.sigmoid_focal_crossentropy
                }
            )

    def _load_map(self, map_path: str, normalise: bool = True):
        self.map: gemmi.Ccp4Map = gemmi.read_ccp4_map(map_path)
        self.raw_grid: gemmi.FloatGrid = self.map.grid

        if normalise:
            self.raw_grid.normalize()

    def _load_structure(self, pdb_code: str):
        self.structure = data.import_pdb(pdb_code)
        if not self.structure:
            raise RuntimeError

    def _load_cache(self, cache_path: str):
        predicted_map_path = os.path.join(cache_path, self.predicted_name)
        sugar_map_path = os.path.join(cache_path, self.sugar_name)
        base_map_path = os.path.join(cache_path, self.base_name)
        phos_map_path = os.path.join(cache_path, self.phos_name)

        self.predicted_map = np.load(predicted_map_path)
        self.sugar_map = np.load(sugar_map_path)
        self.base_map = np.load(base_map_path)
        self.phos_map = np.load(phos_map_path)
        logging.info("Cache loaded")

    def _save_cache(self, cache_path: str):
        predicted_map_path = os.path.join(cache_path, self.predicted_name)
        sugar_map_path = os.path.join(cache_path, self.sugar_name)
        base_map_path = os.path.join(cache_path, self.base_name)
        phos_map_path = os.path.join(cache_path, self.phos_name)

        if not os.path.isdir(cache_path):
            os.makedirs(cache_path)

        np.save(predicted_map_path, self.predicted_map)
        np.save(sugar_map_path, self.sugar_map)
        np.save(base_map_path, self.base_map)
        np.save(phos_map_path, self.phosphate_map)

        logging.info("Saving to cache.")

    def _interpolate_grid(self, grid_spacing: float = 0.7):
        box: gemmi.PositionBox = utils.get_bounding_box(self.raw_grid)
        size: gemmi.Position = box.get_size()

        logging.info(f"Raw unit cell is : {self.raw_grid.unit_cell}")

        logging.debug(f"Box size : {size}")

        num_x = int(size.x / grid_spacing)
        num_y = int(size.y / grid_spacing)
        num_z = int(size.z / grid_spacing)

        logging.debug(f"Num x,y,z: {num_x}, {num_y}, {num_z}")

        array = np.zeros((num_x, num_y, num_z), dtype=np.float32)
        scale = gemmi.Mat33(
            [[grid_spacing, 0, 0], [0, grid_spacing, 0], [0, 0, grid_spacing]]
        )

        self.transform: gemmi.Transform = gemmi.Transform(scale, box.minimum)
        self.raw_grid.interpolate_values(array, self.transform)
        cell: gemmi.UnitCell = gemmi.UnitCell(size.x, size.y, size.z, 90, 90, 90)
        self.interpolated_grid = gemmi.FloatGrid(array, cell)
        logging.debug(f"Interpolated grid (numpy) shape: {array.shape}")
        logging.debug(f"Interpolated grid (gemmi) shape: {self.interpolated_grid}")

    def _calculate_translations(self, overlap: int = 32):
        logging.info("Calculating translations")
        logging.debug(
            f"Interpolated grid unit cell : {self.interpolated_grid.unit_cell}"
        )
        overlap_na: float = (self.interpolated_grid.array.shape[0] // overlap) + 1
        overlap_nb: float = (self.interpolated_grid.array.shape[1] // overlap) + 1
        overlap_nc: float = (self.interpolated_grid.array.shape[2] // overlap) + 1

        logging.debug(f"na, nb, nc: {self.na}, {self.nb}, {self.nc}")

        for x in range(int(overlap_na)):
            for y in range(int(overlap_nb)):
                for z in range(int(overlap_nc)):
                    self.translation_list.append(
                        [x * overlap, y * overlap, z * overlap]
                    )

        self.na: float = (self.interpolated_grid.array.shape[0] // 32) + 1
        self.nb: float = (self.interpolated_grid.array.shape[1] // 32) + 1
        self.nc: float = (self.interpolated_grid.array.shape[2] // 32) + 1

        logging.debug(f"Translation list size: {len(self.translation_list)}")

    def _predict(self, raw_values: bool = True):
        logging.info("Predicting map")
        
        predicted_map = np.zeros(
            (int(32 * self.na) + 16, int(32 * self.nb) + 16, int(32 * self.nc) + 16), np.float32
        )
        count_map = np.zeros(
            (int(32 * self.na) + 16, int(32 * self.nb) + 16, int(32 * self.nc) + 16), np.float32
        )

        for translation in tqdm(self.translation_list, len(self.translation_list)):
            x, y, z = translation
            # print(x, y, z)

            sub_array = np.array(
                self.interpolated_grid.get_subarray(
                    start=translation, shape=[32, 32, 32]
                )
            ).reshape(1, 32, 32, 32, 1)

            predicted_sub = self.model.predict(sub_array, verbose=0).squeeze()
            arg_max = np.argmax(predicted_sub, axis=-1)
            
            # Taken from https://github.com/paulsbond/densitydensenet/blob/main/predict.py
            if raw_values:
               predicted_map[x : x + 32, y : y + 32, z : z + 32] += predicted_sub[:,:,:, 1]
            else:
                predicted_map[x : x + 32, y : y + 32, z : z + 32] += arg_max
            count_map[x : x + 32, y : y + 32, z : z + 32] += 1


        logging.debug(f"Predicted map shape: {predicted_map.shape}")
        self.predicted_map = predicted_map / count_map

    def _generate_sugar_map(self):
        logging.info("Generating sugar map")
        interpolated_array: np.ndarray = self.predicted_map
        sugar_atoms = ["C1'", "C2'", "C3'", "C4'", "C5'", "O2'", "O3'", "O4'", "O5'"]
        sugar_neigbour_search = self._intitialise_atom_search(self.structure, sugar_atoms)
        sugar_map = np.zeros(interpolated_array.shape, dtype=np.float32)

        for i in range(interpolated_array.shape[0]):
            for j in range(interpolated_array.shape[1]):
                for k in range(interpolated_array.shape[2]):
                    index_pos = gemmi.Vec3(i, j, k)
                    position = gemmi.Position(self.transform.apply(index_pos))

                    any_sugars = sugar_neigbour_search.find_atoms(
                        position, "\0", radius=3
                    )
                    sugar_mask = 1.0 if len(any_sugars) > 1 else 0.0
                    sugar_map[i][j][k] = sugar_mask

        logging.debug(f"Sugar map shape: {sugar_map.shape}")
        self.sugar_map = sugar_map
    
    def _generate_phosphate_map(self):
        logging.info("Generating phosphate map")
        interpolated_array: np.ndarray = self.predicted_map

        phosphate_atoms = ["P", "OP1", "OP2", "O5'", "O3'"]

        phosphate_neigbour_search = self._intitialise_atom_search(self.structure, phosphate_atoms)
        phosphate_map = np.zeros(interpolated_array.shape, dtype=np.float32)

        for i in range(interpolated_array.shape[0]):
            for j in range(interpolated_array.shape[1]):
                for k in range(interpolated_array.shape[2]):
                    index_pos = gemmi.Vec3(i, j, k)
                    position = gemmi.Position(self.transform.apply(index_pos))

                    any_sugars = phosphate_neigbour_search.find_atoms(
                        position, "\0", radius=3
                    )
                    sugar_mask = 1.0 if len(any_sugars) > 1 else 0.0
                    phosphate_map[i][j][k] = sugar_mask

        logging.debug(f"Phosphate map shape: {phosphate_map.shape}")
        self.phosphate_map = phosphate_map

    def _generate_base_map(self):
        logging.info("Generating base map")
        interpolated_array: np.ndarray = self.predicted_map

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

        base_neigbour_search = self._intitialise_atom_search(self.structure, base_atoms)
        base_map = np.zeros(interpolated_array.shape, dtype=np.float32)

        for i in range(interpolated_array.shape[0]):
            for j in range(interpolated_array.shape[1]):
                for k in range(interpolated_array.shape[2]):
                    index_pos = gemmi.Vec3(i, j, k)
                    position = gemmi.Position(self.transform.apply(index_pos))

                    any_sugars = base_neigbour_search.find_atoms(
                        position, "\0", radius=3
                    )
                    sugar_mask = 1.0 if len(any_sugars) > 1 else 0.0
                    base_map[i][j][k] = sugar_mask

        logging.debug(f"Base map shape: {base_map.shape}")
        self.base_map = base_map

    
    def _intitialise_atom_search(
        self, structure: gemmi.Structure, atom_list: List[str]
    ) -> gemmi.NeighborSearch:
        neigbour_search = gemmi.NeighborSearch(structure[0], structure.cell, 3)

        for n_ch, chain in enumerate(structure[0]):
            for n_res, res in enumerate(chain):
                for n_atom, atom in enumerate(res):
                    if atom.name in atom_list:
                        neigbour_search.add_atom(atom, n_ch, n_res, n_atom)

        return neigbour_search


    def _score_predicted_map(self) -> Scores:
        logging.info("Scoring prediction")
        sugar_map: np.ndarray = self.sugar_map
        phosphate_map: np.ndarray = self.phosphate_map
        predicted_map: np.ndarray = self.predicted_map

        assert sugar_map.shape == predicted_map.shape

        base_scores = Scores()

        for i in range(predicted_map.shape[0]):
            for j in range(predicted_map.shape[1]):
                for k in range(predicted_map.shape[2]):
                    
                    predicted_val = predicted_map[i][j][k] 
                    base_val = self.base_map[i][j][k]

                    if predicted_val == base_val:
                        if predicted_val == 1: 
                            base_scores.positive += 1
                        elif predicted_val == 0: 
                            base_scores.negative += 1
  
                    else: 
                        if base_val == 1: 
                            base_scores.false_negative += 1
                        else:
                            base_scores.false_positive += 1
  


        return base_scores


def get_test_list(test_dir: str) -> List[Tuple[str, str]]:
    return [(x.path, x.name.strip(".map")) for x in os.scandir(test_dir)][:10]


def predict_new_model():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s %(levelname)s - %(message)s"
    )

    model_dir = "./models/interpolated_model_4"

    test_list = get_test_list("data/DNA_test_structures/external_test_maps/map")

    scores = []

    for map_file, pdb_code in tqdm(test_list):
        if len(pdb_code) != 4:
            continue
        test = TestModel(model_dir=model_dir, use_cache=True)

        try:
            test.make_prediction(map_path=map_file, pdb_code=pdb_code)
        except RuntimeError:
            continue

        test.save_score("results")
        scores.append(test.score)

    scores = np.array(scores)

    logging.info(f"Average Score {np.mean(scores)} += {np.std(scores)}")

def test_pdb_files(): 
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    logging.basicConfig(
        level=logging.CRITICAL, format="%(asctime)s %(levelname)s - %(message)s"
    )

    model_path = "models/base_1.5A_model_1.best.hdf5"
    
    map_folder = "data/map"
    pdb_code = "1hr2"
    
    for map_file in tqdm(os.scandir(map_folder), total=len(os.listdir(map_folder))):    
        test = TestModel(model_dir=model_path, use_cache=True)
        test.make_prediction(map_path=map_file, pdb_code=pdb_code)
        test.save_score("results")

        
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s %(levelname)s - %(message)s"
    )

    model_paths = [
        # "models/base_model_1",
        # "models/interpolated_model_2",
        # "models/phos_model_1"
        "models/base_1.5A_model_1.best.hdf5"
    ]
    map_file = "data/raw_maps/1hr2.map"
    pdb_code = "1hr2"
    
    for model in model_paths:    
        test = TestModel(model_dir=model, use_cache=True)
        test.make_prediction(map_path=map_file, pdb_code=pdb_code)
        test.save_score("results")

        # test.save_maps(output_dir="predictions", original=True)
        quit()


if __name__ == "__main__":
    # predict_new_model()
    # main()
    test_pdb_files()
