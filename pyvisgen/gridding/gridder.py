import os
from pathlib import Path

import astropy.constants as const
import h5py
import numpy as np
from tqdm import tqdm

from pyvisgen.fits.data import fits_data
from pyvisgen.gridding.alt_gridder import ms2dirty_python_fast
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

    ###################
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

            # rescaled to level Stokes I
            gridded_data_test /= 2
            save_fft_pair(out, gridded_data_test, truth_amp_phase_test)
    #
    ###################

    size_train = int(size // (1 + conf["train_valid_split"]))
    size_valid = size - size_train
    print(f"Training size: {size_train}, Validation size: {size_valid}")
    bundle_train = int(size_train // conf["bundle_size"])
    bundle_valid = int(size_valid // conf["bundle_size"])

    ###################
    # train
    for i in tqdm(range(bundle_train)):
        i += bundle_test
        uv_data_train, freq_data_train, gridded_data_train, sky_dist_train = open_data(
            fits_files, sky_dist, conf, i
        )

        truth_fft_train = calc_truth_fft(sky_dist_train)

        # sim_real_imag_train = np.array(
        #     (gridded_data_train[:, 0] + 1j * gridded_data_train[:, 1])
        # )
        # dirty_image_train = np.abs(
        #     np.fft.fftshift(
        #         np.fft.fft2(
        #             np.fft.fftshift(sim_real_imag_train, axes=(1, 2)), axes=(1, 2)
        #         ),
        #         axes=(1, 2),
        #     )
        # )

        if conf["amp_phase"]:
            gridded_data_train = convert_amp_phase(gridded_data_train, sky_sim=False)
            truth_amp_phase_train = convert_amp_phase(truth_fft_train, sky_sim=True)
        else:
            gridded_data_train = convert_real_imag(gridded_data_train, sky_sim=False)
            truth_amp_phase_train = convert_real_imag(truth_fft_train, sky_sim=True)

        out = out_path / Path("samp_train" + str(i) + ".h5")

        # rescaled to level Stokes I
        gridded_data_train /= 2
        save_fft_pair(out, gridded_data_train, truth_amp_phase_train)
        train_index_last = i
    #
    ###################

    ###################
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

        # rescaled to level Stokes I
        gridded_data_valid /= 2
        save_fft_pair(out, gridded_data_valid, truth_amp_phase_valid)
    #
    ###################


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


def calc_truth_fft(sky_dist):
    # norm = np.sum(np.sum(sky_dist_test, keepdims=True, axis=1), axis=2)
    # sky_dist_test = np.expand_dims(sky_dist_test, -1) / norm[:, None, None]
    truth_fft = np.fft.fftshift(
        np.fft.fft2(np.fft.fftshift(sky_dist, axes=(1, 2)), axes=(1, 2)), axes=(1, 2)
    )
    return truth_fft


def ducc0_gridding(uv_data, freq_data):
    vis_ = uv_data["DATA"]
    vis = np.array([vis_[:, 0, 0, 0, 0, 0, 0] + 1j * vis_[:, 0, 0, 0, 0, 0, 1]]).T
    vis_compl = np.array([vis_[:, 0, 0, 0, 0, 0, 0] + 1j * vis_[:, 0, 0, 0, 0, 0, 1]]).T
    uu = np.array(uv_data["UU--"], dtype=np.float64)
    uu_compl = np.array(-uv_data["UU--"], dtype=np.float64)
    vv = np.array(uv_data["VV--"], dtype=np.float64)
    vv_compl = np.array(-uv_data["VV--"], dtype=np.float64)
    ww = np.array(uv_data["WW--"], dtype=np.float64)
    ww_compl = np.array(uv_data["WW--"], dtype=np.float64)
    uvw = np.stack([uu, vv, ww]).T
    uvw_compl = np.stack([uu_compl, vv_compl, ww_compl]).T
    uvw *= const.c.value
    uvw_compl *= const.c.value
    # complex conjugated
    uvw = np.append(uvw, uvw_compl, axis=0)
    vis = np.append(vis, vis_compl)

    freq = freq_data[1]
    freq = (freq_data[0]["IF FREQ"] + freq).reshape(-1, 1)[0]
    wgt = np.ones((vis.shape[0], 1))
    mask = None

    wgt[vis == 0] = 0
    if mask is None:
        mask = np.ones(wgt.shape, dtype=np.uint8)
    mask[wgt == 0] = False

    DEG2RAD = np.pi / 180
    # nthreads = 4
    epsilon = 1e-4
    # do_wgridding = False
    # verbosity = 1

    # do_sycl = False  # True
    # do_cng = False  # True

    # ntries = 1

    fov_deg = 0.02  # 1e-5  # 3.3477833333331884e-5

    npixdirty = 64  # get_npixdirty(uvw, freq, fov_deg, mask)
    pixsize = fov_deg / npixdirty * DEG2RAD

    # mintime = 1e300

    grid = ms2dirty_python_fast(
        uvw, freq, vis, npixdirty, npixdirty, pixsize, pixsize, epsilon, False
    )
    grid = np.rot90(np.fft.fftshift(grid))
    # assert grid.shape[0] == 256
    return grid


def grid_data(uv_data, freq_data, conf):
    cmplx = uv_data["DATA"]
    real = np.squeeze(cmplx[..., 0, 0, 0])  # .ravel()
    imag = np.squeeze(cmplx[..., 0, 0, 1])  # .ravel()
    # weight = np.squeeze(cmplx[..., 0, 2])

    freq = freq_data[1]
    IF_bands = (freq_data[0]["IF FREQ"] + freq).reshape(-1, 1)

    u = np.repeat([uv_data["UU--"]], real.shape[1], axis=0)
    v = np.repeat([uv_data["VV--"]], real.shape[1], axis=0)
    u = (u * IF_bands).T.ravel()
    v = (v * IF_bands).T.ravel()

    real = real.ravel()
    imag = imag.ravel()

    samps = np.array(
        [
            np.append(u, -u),
            np.append(v, -v),
            np.append(real, real),
            np.append(imag, -imag),
        ]
    )
    # Generate Mask
    N = conf["grid_size"]  # image size
    fov = (
        conf["fov_size"] * np.pi / (3600 * 180)
    )  # hard code #default 0.00018382, FoV from VLBA 163.7 <- wrong!
    # depends on setting of simulations
    delta = 1 / fov * const.c.value / conf["ref_frequency"]

    bins = np.arange(start=-(N / 2) * delta, stop=(N / 2 + 1) * delta, step=delta)
    if len(bins) - 1 > N:
        bins = np.delete(bins, -1)

    mask, *_ = np.histogram2d(samps[0], samps[1], bins=[bins, bins], density=False)
    mask[mask == 0] = 1

    mask_real, x_edges, y_edges = np.histogram2d(
        samps[0], samps[1], bins=[bins, bins], weights=samps[2], density=False
    )
    mask_imag, x_edges, y_edges = np.histogram2d(
        samps[0], samps[1], bins=[bins, bins], weights=samps[3], density=False
    )

    mask_real /= mask
    mask_imag /= mask

    mask_real = np.rot90(mask_real, 1)
    mask_imag = np.rot90(mask_imag, 1)

    gridded_vis = np.zeros((2, N, N))
    gridded_vis[0] = mask_real
    gridded_vis[1] = mask_imag
    return gridded_vis


def convert_amp_phase(data, sky_sim=False):
    if sky_sim:
        amp = np.abs(data)
        phase = np.angle(data)
        data = np.stack((amp, phase), axis=1)
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

        data = np.stack((real, imag), axis=1)
    else:
        real = data[:, 0]
        imag = data[:, 1]

        data = np.stack((real, imag), axis=1)
    return data


def save_fft_pair(path, x, y, name_x="x", name_y="y"):
    """
    write fft_pairs created in second analysis step to h5 file
    """
    half_image = x.shape[2] // 2
    x = x[:, :, : half_image + 1, :]
    y = y[:, :, : half_image + 1, :]
    with h5py.File(path, "w") as hf:
        hf.create_dataset(name_x, data=x)
        hf.create_dataset(name_y, data=y)
        hf.close()


if __name__ == "__main__":
    create_gridded_data_set(
        "/net/big-tank/POOL/projects/radio/test_rime/create_dataset.toml"
    )
