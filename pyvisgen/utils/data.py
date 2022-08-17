import h5py
import torch
import numpy as np
import re
from pathlib import Path


class data_handler:
    def __init__(self, bundle_paths):
        """
        Save the bundle paths and the number of bundles in one file.
        """
        if bundle_paths == []:
            raise ValueError('No bundles found! Please check the names of your files.')
        self.bundles = bundle_paths
        self.num_img = len(self.open_bundle(self.bundles[0], "x"))

    def __call__(self):
        return print("This is the h5_dataset class.")

    def __len__(self):
        """
        Returns the total number of pictures in this dataset
        """
        return len(self.bundles) * self.num_img

    def __getitem__(self, i):
        x = self.open_image("x", i)
        y = self.open_image("y", i)
        return x, y

    def open_bundle(self, bundle_path, var):
        bundle = h5py.File(bundle_path, "r")
        data = bundle[var]
        return data

    def open_image(self, var, i):
        if isinstance(i, int):
            i = torch.tensor([i])
        elif isinstance(i, np.ndarray):
            i = torch.tensor(i)
        indices, _ = torch.sort(i)
        bundle = torch.div(indices, self.num_img, rounding_mode="floor")
        image = indices - bundle * self.num_img
        bundle_unique = torch.unique(bundle)
        bundle_paths = [
            h5py.File(self.bundles[bundle], "r") for bundle in bundle_unique
        ]
        bundle_paths_str = list(map(str, bundle_paths))
        data = torch.tensor(
            np.array(
                [
                    bund[var][img]
                    for bund, bund_str in zip(bundle_paths, bundle_paths_str)
                    for img in image[
                        bundle == bundle_unique[bundle_paths_str.index(bund_str)]
                    ]
                ]
            )
        )

        if data.shape[0] == 1:
            data = data.squeeze(0)

        return data.float()


def load_data(data_path):
    """
    Load data set from a directory and return it as h5_dataset.

    Parameters
    ----------
    data_path: str
        path to data directory
    mode: str
        specify data set type, e.g. test
    fourier: bool
        use Fourier images as target if True, default is False

    Returns
    -------
    test_ds: h5_dataset
        dataset containing x and y images
    """
    bundle_paths = get_bundles(data_path)
    data = np.sort(
        [path for path in bundle_paths if re.findall("fft_", path.name)]
    )
    print(data)
    # data = sorted(data, key=lambda f: int("".join(filter(str.isdigit, str(f)))))
    # ds = open_data(data)
    return data


def get_bundles(path):
    """
    returns list of bundle paths located in a directory
    """
    data_path = Path(path)
    bundles = np.array([x for x in data_path.iterdir()])
    return bundles


def open_data(path):
    f = h5py.File(path, "r")
    bundle_y = np.array(f["y"])
    return bundle_y
