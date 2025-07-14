import os
from pathlib import Path

import h5py
import numpy as np
from tqdm.auto import tqdm

from pyvisgen.fits.data import fits_data
from pyvisgen.gridding import grid_data
from pyvisgen.utils.config import read_data_set_conf
from pyvisgen.utils.data import load_bundles, open_bundles

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def create_gridded_data_set(config):
    conf = read_data_set_conf(config)
    out_path_fits = Path(conf["out_path_fits"])
    out_path = Path(conf["out_path_gridded"])
    out_path.mkdir(parents=True, exist_ok=True)

    sky_dist = load_bundles(conf["in_path"])
    fits_files = fits_data(out_path_fits)
    size = len(fits_files)
    print(size)

    # test
    if conf["num_test_images"] > 0:
        bundle_test = int(conf["num_test_images"] // conf["bundle_size"])
        size -= conf["num_test_images"]

        for i in tqdm(range(bundle_test)):
            (
                uv_data_test,
                freq_data_test,
                gridded_data_test,
                sky_dist_test,
            ) = open_data(fits_files, sky_dist, conf, i)

            truth_fft_test = calc_truth_fft(sky_dist_test)

            if conf["amp_phase"]:
                gridded_data_test = convert_amp_phase(gridded_data_test, sky_sim=False)
                truth_amp_phase_test = convert_amp_phase(truth_fft_test, sky_sim=True)
            else:
                gridded_data_test = convert_real_imag(gridded_data_test, sky_sim=False)
                truth_amp_phase_test = convert_real_imag(truth_fft_test, sky_sim=True)
            assert gridded_data_test.shape[1] == 2

            out = out_path / Path("samp_test" + str(i) + ".h5")

            save_fft_pair(out, gridded_data_test, truth_amp_phase_test)

    size_train = int(size // (1 + conf["train_valid_split"]))
    size_valid = size - size_train
    print(f"Training size: {size_train}, Validation size: {size_valid}")
    bundle_train = int(size_train // conf["bundle_size"])
    bundle_valid = int(size_valid // conf["bundle_size"])

    # train
    for i in tqdm(range(bundle_train)):
        i += bundle_test
        uv_data_train, freq_data_train, gridded_data_train, sky_dist_train = open_data(
            fits_files, sky_dist, conf, i
        )

        truth_fft_train = calc_truth_fft(sky_dist_train)

        if conf["amp_phase"]:
            gridded_data_train = convert_amp_phase(gridded_data_train, sky_sim=False)
            truth_amp_phase_train = convert_amp_phase(truth_fft_train, sky_sim=True)
        else:
            gridded_data_train = convert_real_imag(gridded_data_train, sky_sim=False)
            truth_amp_phase_train = convert_real_imag(truth_fft_train, sky_sim=True)

        out = out_path / Path("samp_train" + str(i - bundle_test) + ".h5")

        save_fft_pair(out, gridded_data_train, truth_amp_phase_train)
        train_index_last = i

    # valid
    for i in tqdm(range(bundle_valid)):
        i += train_index_last
        uv_data_valid, freq_data_valid, gridded_data_valid, sky_dist_valid = open_data(
            fits_files, sky_dist, conf, i
        )

        truth_fft_valid = calc_truth_fft(sky_dist_valid)

        if conf["amp_phase"]:
            gridded_data_valid = convert_amp_phase(gridded_data_valid, sky_sim=False)
            truth_amp_phase_valid = convert_amp_phase(truth_fft_valid, sky_sim=True)
        else:
            gridded_data_valid = convert_real_imag(gridded_data_valid, sky_sim=False)
            truth_amp_phase_valid = convert_real_imag(truth_fft_valid, sky_sim=True)

        out = out_path / Path("samp_valid" + str(i - train_index_last) + ".h5")

        save_fft_pair(out, gridded_data_valid, truth_amp_phase_valid)


def open_data(fits_files, sky_dist, conf, i):
    sky_sim_bundle_size = len(open_bundles(sky_dist[0]))
    uv_data = [
        fits_files.get_uv_data(n).copy()
        for n in np.arange(
            i * sky_sim_bundle_size, (i * sky_sim_bundle_size) + sky_sim_bundle_size
        )
    ]
    freq_data = np.array(
        [
            fits_files.get_freq_data(n)
            for n in np.arange(
                i * sky_sim_bundle_size, (i * sky_sim_bundle_size) + sky_sim_bundle_size
            )
        ],
        dtype="object",
    )
    gridded_data = np.array(
        [grid_data(data, freq, conf).copy() for data, freq in zip(uv_data, freq_data)]
    )
    bundle = np.floor_divide(i * sky_sim_bundle_size, sky_sim_bundle_size)
    gridded_truth = np.array(
        [
            open_bundles(sky_dist[bundle])[n]
            for n in np.arange(
                i * sky_sim_bundle_size - bundle * sky_sim_bundle_size,
                (i * sky_sim_bundle_size)
                + sky_sim_bundle_size
                - bundle * sky_sim_bundle_size,
            )
        ]
    )
    return uv_data, freq_data, gridded_data, gridded_truth


def save_fft_pair(path, x, y, name_x="x", name_y="y"):
    """
    write fft_pairs created in second analysis step to h5 file
    """
    half_image = x.shape[2] // 2
    x = x[:, :, : half_image + 1, :]
    y = y[:, :, : half_image + 1, :]

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
