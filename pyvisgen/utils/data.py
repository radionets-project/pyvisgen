import h5py
import numpy as np
import re
from pathlib import Path


def load_bundles(data_path):
    bundle_paths = get_bundles(data_path)
    bundles = np.sort([path for path in bundle_paths if re.findall("fft_", path.name)])
    return bundles


def get_bundles(path):
    data_path = Path(path)
    bundles = np.array([x for x in data_path.iterdir()])
    return bundles


def open_bundles(path):
    f = h5py.File(path, "r")
    bundle_y = np.array(f["y"])
    return bundle_y
