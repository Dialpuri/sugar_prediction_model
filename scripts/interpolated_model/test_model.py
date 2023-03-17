import logging
import os
from typing import List
import tensorflow as tf
import tensorflow_addons as tfa
import utils
import import_data as data
import numpy as np
import gemmi


class TestModel:
    CACHE_PATH = "cache"
    PREDICTED_NAME = "predicted.npy"
    SUGAR_NAME = "sugar.npy"

    def __init__(self, model_dir: str, use_cache: bool = True):
        self.use_cache: bool = use_cache
        self.model_dir: str = model_dir

        self.score = 0
        self.predicted_map: np.ndarray = None
        self.sugar_map: np.ndarray = None

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

    def make_prediction(self, map_path: str, pdb_code: str):

        cache_path = os.path.join(self.CACHE_PATH, pdb_code)

        if not self.use_cache or not os.path.isdir(cache_path):
            logging.info("No cache found")
            self._load_model()
            self._load_map(map_path=map_path)
            self._load_structure(pdb_code=pdb_code)
            self._interpolate_grid()
            self._calculate_translations()
            self._predict()
            self._generate_sugar_map()
            self.score = self._score_predicted_map()
            self._save_cache(cache_path)
        else:
            self._load_cache(cache_path)
            self.score = self._score_predicted_map()

    def _load_model(self):
        self.model = tf.keras.models.load_model(self.model_dir, custom_objects={
            'sigmoid_focal_crossentropy': tfa.losses.sigmoid_focal_crossentropy})

    def _load_map(self, map_path: str, normalise: bool = True):
        self.map: gemmi.Ccp4Map = gemmi.read_ccp4_map(map_path)
        self.raw_grid: gemmi.FloatGrid = self.map.grid

        if normalise:
            self.raw_grid.normalize()

    def _load_structure(self, pdb_code: str):
        self.structure = data.import_pdb(pdb_code)

    def _load_cache(self, cache_path: str):

        predicted_map_path = os.path.join(cache_path, self.PREDICTED_NAME)
        sugar_map_path = os.path.join(cache_path, self.SUGAR_NAME)

        self.predicted_map = np.load(predicted_map_path)
        self.sugar_map = np.load(sugar_map_path)
        logging.info("Cache loaded")

    def _save_cache(self, cache_path: str):
        predicted_map_path = os.path.join(cache_path, self.PREDICTED_NAME)
        sugar_map_path = os.path.join(cache_path, self.SUGAR_NAME)

        os.mkdir(cache_path)

        np.save(predicted_map_path, self.predicted_map)
        np.save(sugar_map_path, self.sugar_map)
        logging.info("Saving to cache.")

    def _interpolate_grid(self, grid_spacing: float = 0.7):

        box: gemmi.PositionBox = utils.get_bounding_box(self.raw_grid)
        size: gemmi.Position = box.get_size()

        num_x = -(-int(size.x / grid_spacing) // 16 * 16)
        num_y = -(-int(size.y / grid_spacing) // 16 * 16)
        num_z = -(-int(size.z / grid_spacing) // 16 * 16)

        array = np.zeros((num_x, num_y, num_z), dtype=np.float32)
        scale = gemmi.Mat33(
            [[grid_spacing, 0, 0], [0, grid_spacing, 0], [0, 0, grid_spacing]]
        )

        self.transform: gemmi.Transform = gemmi.Transform(scale, box.minimum)
        self.raw_grid.interpolate_values(array, self.transform)
        cell: gemmi.UnitCell = gemmi.UnitCell(size.x, size.y, size.z, 90, 90, 90)
        self.interpolated_grid = gemmi.FloatGrid(array, cell)

    def _calculate_translations(self, overlap: int = 32):
        logging.info("Calculating translations")
        self.na: float = (self.interpolated_grid.unit_cell.a // overlap) + 1
        self.nb: float = (self.interpolated_grid.unit_cell.b // overlap) + 1
        self.nc: float = (self.interpolated_grid.unit_cell.c // overlap) + 1

        for x in range(int(self.na)):
            for y in range(int(self.nb)):
                for z in range(int(self.nc)):
                    self.translation_list.append([x * overlap, y * overlap, z * overlap])

    def _predict(self):
        logging.info("Predicting map")
        predicted_map = np.zeros(self.interpolated_grid.array.shape, np.float32)

        for translation in self.translation_list:
            x, y, z = translation
            sub_array = np.array(
                self.interpolated_grid.get_subarray(start=translation, shape=[32, 32, 32])
            ).reshape(1, 32, 32, 32, 1)

            predicted_sub = self.model.predict(sub_array).squeeze()
            arg_max = np.argmax(predicted_sub, axis=-1)
            predicted_map[x: x + 32, y: y + 32, z: z + 32] += arg_max

        self.predicted_map = predicted_map

    def _generate_sugar_map(self):
        logging.info("Generating sugar map")
        interpolated_array: np.ndarray = self.interpolated_grid.array

        sugar_neigbour_search = self._initialise_sugar_search(self.structure)
        sugar_map = np.zeros(interpolated_array.shape, dtype=np.float32)

        for i in range(interpolated_array.shape[0]):
            for j in range(interpolated_array.shape[1]):
                for k in range(interpolated_array.shape[2]):
                    index_pos = gemmi.Vec3(i, j, k)
                    position = gemmi.Position(self.transform.apply(index_pos))

                    any_sugars = sugar_neigbour_search.find_atoms(position, "\0", radius=3)
                    sugar_mask = 1.0 if len(any_sugars) > 1 else 0.0
                    sugar_map[i][j][k] = sugar_mask

        self.sugar_map = sugar_map

    def _initialise_sugar_search(self, structure: gemmi.Structure) -> gemmi.NeighborSearch:

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

        correct = 0
        wrong = 0

        for i in range(sugar_map.shape[0]):
            for j in range(sugar_map.shape[1]):
                for k in range(sugar_map.shape[2]):
                    predicted_value = predicted_map[i][j][k]
                    sugar_value = sugar_map[i][j][k]

                    if sugar_value == predicted_value:
                        correct += 1
                    else:
                        wrong += 1

        score = correct / (correct + wrong)

        return score


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

    test = TestModel("./models/interpolated_model_2", use_cache=True)
    test.make_prediction("data/DNA_test_structures/external_test_maps/1hr2.map", "1hr2")


if __name__ == "__main__":
    main()
