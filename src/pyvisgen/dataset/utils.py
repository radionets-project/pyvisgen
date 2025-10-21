import os

import h5py
import numpy as np

from pyvisgen.utils.logging import setup_logger

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
LOGGER = setup_logger()


def save_fft_pair(path, x, y, name_x="x", name_y="y"):
    """
    write fft_pairs created in second analysis step to h5 file
    """
    half_image = x.shape[2] // 2
    x = x[:, :, : half_image + 5, :]
    y = y[:, :, : half_image + 5, :]

    test_shapes(x, "x")
    test_shapes(y, "y")

    with h5py.File(path, "w") as hf:
        hf.create_dataset(name_x, data=x)
        hf.create_dataset(name_y, data=y)
        hf.close()


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


def calc_truth_fft(sky_dist):
    truth_fft = np.fft.fftshift(
        np.fft.fft2(np.fft.fftshift(sky_dist, axes=(2, 3)), axes=(2, 3)), axes=(2, 3)
    )

    return truth_fft


def convert_amp_phase(data, sky_sim=False):
    if sky_sim:
        amp = np.abs(data)
        phase = np.angle(data)
        data = np.concatenate((amp, phase), axis=1)
    else:
        test = data[:, 0] + 1j * data[:, 1]
        amp = np.abs(test)
        phase = np.angle(test)
        data = np.stack((amp, phase), axis=1)

    return data


def convert_real_imag(data, sky_sim=False):
    if sky_sim:
        real = data.real
        imag = data.imag

        data = np.concatenate((real, imag), axis=1)
    else:
        real = data[:, 0]
        imag = data[:, 1]

        data = np.stack((real, imag), axis=1)

    return data
