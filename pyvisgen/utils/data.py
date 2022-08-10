import re
import h5py
import numpy as np
from pathlib import Path


class data_handler:
    def __init__(self, data_path):
        """Handle h5 files simulated with radiosim package.

        Parameters
        ----------
        data_path: str
            path to data directory

        """
        self.path = data_path
        self.bundles = self.get_bundles(Path(data_path))
        self.num_skysims_per_file = len(self.get_skysims(self.bundles[0]))
        self.num_skysims = self.num_skysims_per_file * len(self.bundles)

    def __call__(self):
        return print("This is the h5 radiosim data set class.")

    def __len__(self):
        """
        Returns the total number of sky simulations in this dataset
        """
        return self.num_skysims

    def __getitem__(self, i):
        return self.open_simulation(i)

    def open_simulation(self, i):
        if isinstance(i, int):
            i = np.array([i])
        indices = np.sort(i)
        bundle = np.divmod(indices, self.num_skysims_per_file)[0]
        skysim = indices - bundle * self.num_skysims_per_file
        bundle_unique = np.unique(bundle)
        bundle_paths = [self.bundles[bundle] for bundle in bundle_unique]
        bundle_paths_str = list(map(str, bundle_paths))
        sim = np.array(
            [
                self.get_sim(bund, sky)
                for bund, bund_str in zip(bundle_paths, bundle_paths_str)
                for sky in skysim[
                    bundle == bundle_unique[bundle_paths_str.index(bund_str)]
                ]
            ]
        )
        return sim[:, 0], sim[:, 1], sim[:, 2]

    def get_sim(self, bundle, i):
        skysim = self.get_skysims(bundle)[i]
        comps = self.get_comps(bundle)[i]
        comp_list = self.get_comp_lists(bundle)[i]
        return np.array([skysim, comps, comp_list], dtype="object")

    def get_bundles(self, path):
        return np.sort(np.array([x for x in path.iterdir()]))

    def get_skysims(self, bundle):
        f = h5py.File(bundle)
        skysims = [np.array(f[key]) for key in list(f.keys()) if re.findall("sky", key)]
        f.close()
        return skysims

    def get_comps(self, bundle):
        f = h5py.File(bundle)
        comps = [np.array(f[key]) for key in list(f.keys()) if re.findall("comp", key)]
        f.close()
        return comps

    def get_comp_lists(self, bundle):
        f = h5py.File(bundle)
        comp_lists = [
            np.array(f[key]) for key in list(f.keys()) if re.findall("list", key)
        ]
        f.close()
        return comp_lists
