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


class TestModel:
    CACHE_PATH = "cache"
    PREDICTED_NAME = "predicted.npy"
    SUGAR_NAME = "sugar.npy"

    def __init__(self, model_dir: str, use_cache: bool = True):
        self.use_cache: bool = use_cache
        self.model_dir: str = model_dir
        self.model_name: str = model_dir.split("/")[-1]

        self.score = 0
        self.false_score = 0
        self.predicted_map: np.ndarray = None
        self.sugar_map: np.ndarray = None
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
        return f"Test Model({self.pdb_code}, {self.score}, {self.false_score}"

    def __str__(self):
        return f"Test Model of {self.pdb_code} with score: {self.score:.2f} and false score: {self.false_score:.2f} "

    def __le__(self, other):
        return self.score <= other.score

    def __ge__(self, other):
        return self.score >= other.score

    def make_prediction(self, map_path: str, pdb_code: str):
        self.pdb_code = pdb_code
        cache_path = os.path.join(self.CACHE_PATH, pdb_code, self.model_name)

        if not self.use_cache or not os.path.isdir(cache_path):
            logging.info("No cache found")
            self._load_model()
            self._load_map(map_path=map_path)
            self._load_structure(pdb_code=pdb_code)
            self._interpolate_grid()
            self._calculate_translations()
            self._predict()
            self._generate_sugar_map()
            self.score, self.false_score = self._score_predicted_map()
            self._save_cache(cache_path)
        else:
            self._load_cache(cache_path)
            self.score, self.false_score = self._score_predicted_map()

    def save_score(self, output_dir: str):
        model_name = self.model_dir.split("/")[-1]
        output_dir = os.path.join(output_dir, model_name)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        output_path = os.path.join(output_dir, f"{self.pdb_code}.csv")
        with open(output_path, "w") as output_file:
            output_file.write("PDB,Score,FalseScore\n")
            output_file.write(
                f"{self.pdb_code},{self.score:.3f},{self.false_score:.3f}"
            )

    def save_predicted_map(self, output_dir: str, grid_spacing: float = 0.7):
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

    def save_sugar_map(self, output_dir: str, grid_spacing: float = 0.7):
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

    def save_interpolated_map(self, output_dir: str, grid_spacing: float = 0.7):
        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = self.interpolated_grid
        ccp4.update_ccp4_header()

        output_path = os.path.join(output_dir, f"{self.pdb_code}_interpolated.map")
        ccp4.write_ccp4_map(output_path)

    def save_maps(self, output_dir: str):
        output_dir = os.path.join(output_dir, self.pdb_code, self.model_name)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        self.save_predicted_map(output_dir=output_dir)
        self.save_sugar_map(output_dir=output_dir)
        self.save_interpolated_map(output_dir=output_dir)

    def _load_model(self):
        self.model = tf.keras.models.load_model(
            self.model_dir,
            custom_objects={
                "sigmoid_focal_crossentropy": tfa.losses.sigmoid_focal_crossentropy
            },
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
        predicted_map_path = os.path.join(cache_path, self.PREDICTED_NAME)
        sugar_map_path = os.path.join(cache_path, self.SUGAR_NAME)

        self.predicted_map = np.load(predicted_map_path)
        self.sugar_map = np.load(sugar_map_path)
        logging.info("Cache loaded")

    def _save_cache(self, cache_path: str):
        predicted_map_path = os.path.join(cache_path, self.PREDICTED_NAME)
        sugar_map_path = os.path.join(cache_path, self.SUGAR_NAME)

        if not os.path.isdir(cache_path):
            os.makedirs(cache_path)

        np.save(predicted_map_path, self.predicted_map)
        np.save(sugar_map_path, self.sugar_map)
        logging.info("Saving to cache.")

    def _interpolate_grid(self, grid_spacing: float = 0.7):
        box: gemmi.PositionBox = utils.get_bounding_box(self.raw_grid)
        size: gemmi.Position = box.get_size()

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
        self.na: float = (self.interpolated_grid.array.shape[0] // overlap) + 1
        self.nb: float = (self.interpolated_grid.array.shape[1] // overlap) + 1
        self.nc: float = (self.interpolated_grid.array.shape[2] // overlap) + 1

        logging.debug(f"na, nb, nc: {self.na}, {self.nb}, {self.nc}")

        for x in range(int(self.na)):
            for y in range(int(self.nb)):
                for z in range(int(self.nc)):
                    self.translation_list.append(
                        [x * overlap, y * overlap, z * overlap]
                    )

        logging.debug(f"Translation list size: {len(self.translation_list)}")

    def _predict(self):
        logging.info("Predicting map")
        predicted_map = np.zeros(
            (int(32 * self.na), int(32 * self.nb), int(32 * self.nc)), np.float32
        )

        for translation in self.translation_list:
            x, y, z = translation

            # logging.debug(f"{x},{y},{z} -> {x+32},{y+32},{z+32}")

            sub_array = np.array(
                self.interpolated_grid.get_subarray(
                    start=translation, shape=[32, 32, 32]
                )
            ).reshape(1, 32, 32, 32, 1)

            predicted_sub = self.model.predict(sub_array, verbose=0).squeeze()
            arg_max = np.argmax(predicted_sub, axis=-1)

            predicted_map[x : x + 32, y : y + 32, z : z + 32] += arg_max

        logging.debug(f"Predicted map shape: {predicted_map.shape}")
        self.predicted_map = predicted_map

    def _generate_sugar_map(self):
        logging.info("Generating sugar map")
        interpolated_array: np.ndarray = self.predicted_map

        sugar_neigbour_search = self._initialise_sugar_search(self.structure)
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

    def _initialise_sugar_search(
        self, structure: gemmi.Structure
    ) -> gemmi.NeighborSearch:
        sugar_neigbour_search = gemmi.NeighborSearch(structure[0], structure.cell, 3)
        sugar_atoms = ["C1'", "C2'", "C3'", "C4'", "C5'", "O2'", "O3'", "O4'", "O5'"]

        for n_ch, chain in enumerate(structure[0]):
            for n_res, res in enumerate(chain):
                for n_atom, atom in enumerate(res):
                    if atom.name in sugar_atoms:
                        sugar_neigbour_search.add_atom(atom, n_ch, n_res, n_atom)

        return sugar_neigbour_search

    def _score_predicted_map(self) -> float:
        logging.info("Scoring prediction")
        sugar_map: np.ndarray = self.sugar_map
        predicted_map: np.ndarray = self.predicted_map

        assert sugar_map.shape == predicted_map.shape

        correct = 0
        wrong = 0

        false_positive = 0

        for i in range(sugar_map.shape[0]):
            for j in range(sugar_map.shape[1]):
                for k in range(sugar_map.shape[2]):
                    predicted_value = predicted_map[i][j][k]
                    sugar_value = sugar_map[i][j][k]

                    if predicted_value == 1 and sugar_value == 0:
                        false_positive += 1

                    if sugar_value == predicted_value:
                        correct += 1
                    else:
                        wrong += 1

        score = correct / (correct + wrong)
        false_score = false_positive / (correct + wrong)

        return score, false_score


def get_test_list(test_dir: str) -> List[Tuple[str, str]]:
    return [(x.path, x.name.strip(".map")) for x in os.scandir(test_dir)][:10]


def predict_new_model():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s"
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


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s"
    )

    model_dir = "./models/interpolated_model_3"
    map_file = "data/DNA_test_structures/external_test_maps/map/1hr2.map"
    pdb_code = "1hr2"
    test = TestModel(model_dir=model_dir, use_cache=True)
    test.make_prediction(map_path=map_file, pdb_code=pdb_code)
    print(test)
    # test.save_maps(output_dir="predictions")


if __name__ == "__main__":
    # predict_new_model()
    main()
