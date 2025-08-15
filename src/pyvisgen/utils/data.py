from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
from natsort import natsorted

if TYPE_CHECKING:
    from typing import String, Union

__all__ = ["load_bundles", "open_bundles"]


def load_bundles(data_path: Union(String, Path), dataset_type: String = "") -> list:
    """Loads bundle paths, filters for HDF5 files, and
    returns them in a naturally ordered list.

    Parameters
    ----------
    data_path : str or Path
        Path to the directory containing the HDF5 files.
    dataset_type : str, optional
        Type of the dataset to filter, e.g. 'train', 'valid', or 'test'.

    Returns
    -------
    bundles : list
        Naturally ordered list containing paths to HDF5 files.
    """
    if isinstance(data_path, str):
        data_path = Path(data_path)

    bundles = natsorted(list(data_path.glob(f"*{dataset_type}*.h5")))

    return bundles


def open_bundles(path: str | Path, key: str = "y") -> np.array:
    """Opens a bundle HDF5 file.

    Parameters
    ----------
    path : str or Path
        Path to the bundle HDF5 file.

    Returns
    -------
    bundle_y : :func:`~numpy.array`
        :func:`~numpy.array` containing data from
        the bundle file.
    """
    f = h5py.File(path, "r")
    bundle_y = np.array(f[key])

    return bundle_y
