import re
from pathlib import Path

import h5py
import numpy as np
from natsort import natsorted

__all__ = ["load_bundles", "get_bundles", "open_bundles"]


def load_bundles(data_path: str | Path) -> list:
    """Loads bundle paths, filters for HDF5 files, and
    returns them in a naturally ordered list.

    Parameters
    ----------
    data_path : str or Path
        Path to the directory containing the HDF5 files.

    Returns
    -------
    bundles : list
        Naturally ordered list containing paths to HDF5 files.
    """
    bundle_paths = get_bundles(data_path)
    bundles = natsorted([path for path in bundle_paths if re.findall(".h5", path.name)])

    return bundles


def get_bundles(path: str | Path) -> np.array:
    """Finds all files located in a given directory.

    Parameters
    ----------
    path : str or Path
        Path to the directory containing the bundle files.

    Returns
    -------
    bundles : :class:`~numpy.ndarray`
        :class:`~numpy.ndarray` containing paths to the bundle
        files.
    """
    data_path = Path(path)
    bundles = np.array([x for x in data_path.iterdir()])

    return bundles


def open_bundles(path: str | Path, key: str = "y") -> np.array:
    """Opens a bundle HDF5 file.

    Parameters
    ----------
    path : str or Path
        Path to the bundle HDF5 file.

    Returns
    -------
    bundle_y : :class:`~numpy.ndarray`
        :class:`~numpy.ndarray` containing data from
        the bundle file.
    """
    f = h5py.File(path, "r")
    bundle_y = np.array(f[key])

    return bundle_y
