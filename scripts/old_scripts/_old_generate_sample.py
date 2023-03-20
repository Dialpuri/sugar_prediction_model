
def _generate_sample(filter_: str):
    map_list = get_map_list(filter_)

    current_pdb_code = ""

    for map_path in map_list:

        pdb_code = map_path.split("/")[-1].split(".")[0].strip(filter_).strip("_")

        if pdb_code != current_pdb_code:
            print(f"[PDB]: Changed from {current_pdb_code}->{pdb_code}")
            current_pdb_code = pdb_code

        structure = data.import_pdb(pdb_code)

        neigbour_search, sugar_neigbour_search, phosphate_neigbour_search, base_neigbour_search = _initialise_neighbour_search(
            structure)

        map_ = gemmi.read_ccp4_map(map_path).grid

        map_.normalize()

        a = map_.unit_cell.a
        b = map_.unit_cell.b
        c = map_.unit_cell.c

        overlap = 8

        na = (a // overlap) + 1
        nb = (b // overlap) + 1
        nc = (c // overlap) + 1

        translation_list = []

        for x in range(int(na)):
            for y in range(int(nb)):
                for z in range(int(nc)):
                    translation_list.append([x * overlap, y * overlap, z * overlap])

        for translation in translation_list:
            sub_array = np.array(map_.get_subarray(start=translation, shape=[32, 32, 32]))
            output_grid = np.zeros((32, 32, 32, 3))

            for i, x in enumerate(sub_array):
                for j, y in enumerate(x):
                    for k, z in enumerate(y):
                        position = gemmi.Position(translation[0] + i, translation[1] + j, translation[2] + k)

                        any_atom = neigbour_search.find_atoms(position, "\0", radius=params.ns_radius)

                        any_bases = base_neigbour_search.find_atoms(position, "\0", radius=params.ns_radius)
                        any_sugars = sugar_neigbour_search.find_atoms(position, "\0", radius=params.ns_radius)
                        any_phosphate = phosphate_neigbour_search.find_atoms(position, "\0", radius=params.ns_radius)

                        mask = 1 if len(any_atom) > 1 else 0
                        base_mask = 1 if len(any_bases) > 1 else 0
                        sugar_mask = 1 if len(any_sugars) > 1 else 0
                        phosphate_mask = 1 if len(any_phosphate) > 1 else 0

                        # encoded_mask = tf.one_hot(mask, depth=2)

                        encoded_mask = [sugar_mask, phosphate_mask, base_mask]

                        output_grid[i][j][k] = encoded_mask

            mask = output_grid.reshape((32, 32, 32, 3))
            original = sub_array.reshape((32, 32, 32, 1))

            if (mask == 1).sum() > 5_000:
                # print("Returning 1")
                yield original, mask
