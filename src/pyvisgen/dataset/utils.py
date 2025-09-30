import os
from pathlib import Path

import h5py
import numpy as np
import torch

from pyvisgen.utils.logging import setup_logger

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
LOGGER = setup_logger()


def save_h5(
    path: str | Path,
    x: np.ndarray,
    y: np.ndarray,
    overlap: int,
    name_x: str = "X",
    name_y: str = "y",
):
    """
    write fft_pairs created in second analysis step to h5 file
    """
    half_image = x.shape[2] // 2
    print(x.shape)
    print(y.shape)
    x = x[:, :, : half_image + overlap, :]
    y = y[:, :, : half_image + overlap, :]

    test_shapes(x, "X")
    test_shapes(y, "y")

    with h5py.File(path, "w") as hf:
        hf.create_dataset(name_x, data=x)
        hf.create_dataset(name_y, data=y)
        hf.close()


def save_pt(
    path: str | Path,
    x: np.ndarray,
    y: np.ndarray,
    overlap: int,
    data_type: str,
    name_x: str = "X",
    name_y: str = "y",
):
    """
    write fft_pairs created in second analysis step to h5 file
    """

    if data_type not in ["amp_phase", "real_imag"]:
        raise ValueError(
            f"The given data type is invalid! Only one of {['amp_phase', 'real_imag']} is allowed!"
        )

    half_image = x.shape[2] // 2
    x = torch.from_numpy(x)[:, :, : half_image + overlap, :]

    x = x[:, 0] + 1j * x[:, 1]

    y = torch.from_numpy(y)[:, :, : half_image + overlap, :]

    test_shapes(x, "X")
    test_shapes(y, "y")

    torch.save(obj={"SIM": x.to_sparse(), "TRUTH": y, "TYPE": data_type}, f=path)


def test_shapes(array, name):
    if array.shape[1] != 2:
        raise ValueError(
            f"Expected array {name} axis 1 to be 2 but got "
            f"{array.shape} with axis 1: {array.shape[1]}!"
        )

    if len(array.shape) != 4:
        raise ValueError(
            f"Expected array {name} shape to be of len 4 but got "
            f"{array.shape} with len {len(array.shape)}!"
        )


def calc_truth_fft(sky_dist: np.ndarray):
    return np.fft.fftshift(
        np.fft.fft2(np.fft.fftshift(sky_dist, axes=(2, 3)), axes=(2, 3)),
        axes=(2, 3),
    )


def convert_amp_phase(data: np.ndarray, sky_sim: bool = False):
    if sky_sim:
        return (np.abs(data) + 1j * np.angle(data))[..., : data.shape[-1] // 2, :]
    else:
        test = data[:, 0] + 1j * data[:, 1]
        return (np.abs(test) + 1j * np.angle(test))[..., : test.shape[-1] // 2, :]


def convert_real_imag(data: torch.Tensor, sky_sim: bool = False):
    if sky_sim:
        return data[..., : data.shape[-1] // 2, :]
    else:
        return (data[:, 0] + 1j * data[:, 1])[..., : data.shape[-1] // 2, :]
