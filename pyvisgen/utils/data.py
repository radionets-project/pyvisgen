import re
import h5py
import numpy as np
from pathlib import Path


def get_data_paths(path):
    data_dir = Path(path)
    bundles = np.array([x for x in data_dir.iterdir()])
    data_paths = np.sort([path for path in bundles if re.findall("source_", path.name)])
    return data_paths


class h5_sky_distributions:
    def __init__(self, bundle_paths):
        """
        Save the bundle paths and the number of bundles in one file.
        """
        self.bundles = bundle_paths
        self.num_img = len(self.open_bundle(self.bundles[0])) // 3

    def __call__(self):
        return print("This is the h5_sky_distributions class.")

    def __len__(self):
        """
        Returns the total number of images in this dataset.
        """
        return len(self.bundles) * self.num_img

    def __getitem__(self, i):
        sky = self.open_image("sky", i)
        return sky

    def open_bundle(self, bundle_path):
        bundle = h5py.File(bundle_path, "r")
        return bundle

    def open_image(self, var, i):
        if isinstance(i, int):
            i = np.array([i])
        indices = np.sort(i)
        bundle = indices // self.num_img
        image = indices - bundle * self.num_img
        bundle_unique = np.unique(bundle)
        bundle_files = [
            h5py.File(self.bundles[bundle], "r") for bundle in bundle_unique
        ]
        bundle_files_str = list(map(str, bundle_files))
        data = np.array(
            [
                bund["sky" + str(img)]
                for bund, bund_str in zip(bundle_files, bundle_files_str)
                for img in image[
                    bundle == bundle_unique[bundle_files_str.index(bund_str)]
                ]
            ]
        )
        return data
