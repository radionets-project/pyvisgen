import h5py
import numpy as np
import re
from pathlib import Path
from natsort import natsorted


def load_bundles(data_path):
    bundle_paths = get_bundles(data_path)
    bundles = natsorted(
        [path for path in bundle_paths if re.findall("skies_", path.name)]
    )
    return bundles


def get_bundles(path):
    data_path = Path(path)
    bundles = np.array([x for x in data_path.iterdir()])
    return bundles


def open_bundles(path):
    f = h5py.File(path, "r")
    bundle_y = np.array(f["y"])
    return bundle_y
