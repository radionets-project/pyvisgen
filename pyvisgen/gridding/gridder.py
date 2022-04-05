import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from radiosim.data import radiosim_data
from pyvisgen.fits.data import fits_data
from pyvisgen.utils.config import read_data_set_conf
from radionets.dl_framework.data import save_fft_pair


def create_gridded_data_set(config):
    conf = read_data_set_conf(config)
    out_path_fits = Path(conf["out_path"])
    out_path = out_path_fits.parent / "gridded_data_large/"
    ######
    # out_path = Path(
    #     "/net/big-tank/POOL/projects/radio/test_rime/build_test/gridded_data"
    # )
    # out_path_fits = Path(
    #     "/net/big-tank/POOL/projects/radio/test_rime/build_test/uvfits"
    # )
    ######
    out_path.mkdir(parents=True, exist_ok=True)

    # conf[
    #     "in_path"
    # ] = "/net/big-tank/POOL/projects/radio/test_rime/build_test/sky_simulations"
    sky_dist = radiosim_data(conf["in_path"])
    fits_files = fits_data(out_path_fits)
    size = len(fits_files)
    print(size)
    # num_bundles = int(size // conf["bundle_size"])

    ###################
    # test
    if conf["num_test_images"] > 0:
        bundle_test = int(conf["num_test_images"] // conf["bundle_size"])
        size -= conf["num_test_images"]

        for i in tqdm(range(bundle_test)):
            uv_data_test = np.array(
                [
                    fits_files.get_uv_data(n)
                    for n in np.arange(
                        i * conf["bundle_size"],
                        (i * conf["bundle_size"]) + conf["bundle_size"],
                    )
                ],
                dtype="object",
            )
            gridded_data_test = np.array(
                [grid_data(data, conf) for data in tqdm(uv_data_test)]
            )
            sky_dist_test = np.array(
                [
                    cv2.resize(
                        sky_dist[int(n)][0][0], (conf["grid_size"], conf["grid_size"])
                    )
                    for n in np.arange(
                        i * conf["bundle_size"],
                        (i * conf["bundle_size"]) + conf["bundle_size"],
                    )
                ]
            )

            # norm = np.sum(np.sum(sky_dist_test, keepdims=True, axis=1), axis=2)
            # sky_dist_test = np.expand_dims(sky_dist_test, -1) / norm[:, None, None]
            truth_fft_test = np.fft.fftshift(
                np.fft.fft2(
                    np.fft.fftshift(sky_dist_test / 1e3, axes=(1, 2)), axes=(1, 2)
                ),
                axes=(1, 2),
            )

            # sim_real_imag_test = np.array(
            #     (gridded_data_test[:, 0] + 1j * gridded_data_test[:, 1])
            # )
            # dirty_image_test = np.abs(
            #     np.fft.fftshift(
            #         np.fft.fft2(
            #             np.fft.fftshift(sim_real_imag_test, axes=(1, 2)), axes=(1, 2)
            #         ),
            #         axes=(1, 2),
            #     )
            # )

            if conf["amp_phase"]:
                gridded_data_test = convert_amp_phase(gridded_data_test)
                truth_amp_phase_test = convert_amp_phase(truth_fft_test, sky_sim=True)

            out = out_path / Path("samp_test" + str(i) + ".h5")
            save_fft_pair(out, gridded_data_test, truth_amp_phase_test)
        #
        #########

    size_valid = conf["train_valid_split"] * size
    size_train = size - size_valid
    bundle_train = int(size_train // conf["bundle_size"])
    bundle_valid = int(size_valid // conf["bundle_size"])

    for i in tqdm(range(bundle_train)):
        i += bundle_test
        uv_data_train = np.array(
            [
                fits_files.get_uv_data(n)
                for n in np.arange(
                    i * conf["bundle_size"],
                    (i * conf["bundle_size"]) + conf["bundle_size"],
                )
            ],
            dtype="object",
        )
        gridded_data_train = np.array(
            [grid_data(data, conf) for data in tqdm(uv_data_train)]
        )
        sky_dist_train = np.array(
            [
                cv2.resize(
                    sky_dist[int(n)][0][0], (conf["grid_size"], conf["grid_size"])
                )
                for n in np.arange(
                    i * conf["bundle_size"],
                    (i * conf["bundle_size"]) + conf["bundle_size"],
                )
            ]
        )

        # norm = np.sum(np.sum(sky_dist_train, keepdims=True, axis=1), axis=2)
        # sky_dist_train = np.expand_dims(sky_dist_train, -1) / norm[:, None, None]
        truth_fft_train = np.fft.fftshift(
            np.fft.fft2(
                np.fft.fftshift(sky_dist_train / 1e3, axes=(1, 2)), axes=(1, 2)
            ),
            axes=(1, 2),
        )

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
            gridded_data_train = convert_amp_phase(gridded_data_train)
            truth_amp_phase_train = convert_amp_phase(truth_fft_train, sky_sim=True)

        out = out_path / Path("samp_train" + str(i) + ".h5")
        save_fft_pair(out, gridded_data_train, truth_amp_phase_train)
        train_index_last = i

    for i in tqdm(range(bundle_valid)):
        i += train_index_last
        uv_data_valid = np.array(
            [
                fits_files.get_uv_data(n)
                for n in np.arange(
                    i * conf["bundle_size"],
                    (i * conf["bundle_size"]) + conf["bundle_size"],
                )
            ],
            dtype="object",
        )
        gridded_data_valid = np.array(
            [grid_data(data, conf) for data in tqdm(uv_data_valid)]
        )
        sky_dist_valid = np.array(
            [
                cv2.resize(
                    sky_dist[int(n)][0][0], (conf["grid_size"], conf["grid_size"])
                )
                for n in np.arange(
                    i * conf["bundle_size"],
                    (i * conf["bundle_size"]) + conf["bundle_size"],
                )
            ]
        )

        # norm = np.sum(np.sum(sky_dist_valid, keepdims=True, axis=1), axis=2)
        # sky_dist_valid = np.expand_dims(sky_dist_valid, -1) / norm[:, None, None]
        truth_fft_valid = np.fft.fftshift(
            np.fft.fft2(
                np.fft.fftshift(sky_dist_valid / 1e3, axes=(1, 2)), axes=(1, 2)
            ),
            axes=(1, 2),
        )

        # sim_real_imag_valid = np.array(
        #     (gridded_data_valid[:, 0] + 1j * gridded_data_valid[:, 1])
        # )
        # dirty_image_valid = np.abs(
        #     np.fft.fftshift(
        #         np.fft.fft2(
        #             np.fft.fftshift(sim_real_imag_valid, axes=(1, 2)), axes=(1, 2)
        #         ),
        #         axes=(1, 2),
        #     )
        # )

        if conf["amp_phase"]:
            gridded_data_valid = convert_amp_phase(gridded_data_valid)
            truth_amp_phase_valid = convert_amp_phase(truth_fft_valid, sky_sim=True)

        out = out_path / Path("samp_valid" + str(i - train_index_last) + ".h5")
        save_fft_pair(out, gridded_data_valid, truth_amp_phase_valid)


def grid_data(uv_data, conf):
    freq = conf["base_freq"]

    cmplx = uv_data["DATA"]
    real = np.squeeze(cmplx[..., 0, 0])
    imag = np.squeeze(cmplx[..., 0, 1])
    # weight = np.squeeze(cmplx[..., 0, 2])

    samps = np.array(
        [
            np.append(uv_data["UU--"] * freq, -uv_data["UU--"] * freq),
            np.append(uv_data["VV--"] * freq, -uv_data["VV--"] * freq),
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
    delta = 1 / fov

    bins = np.arange(start=-(N / 2) * delta, stop=(N / 2 + 1) * delta, step=delta)

    mask, *_ = np.histogram2d(samps[0], samps[1], bins=[bins, bins], normed=False)
    mask[mask == 0] = 1
    mask = np.rot90(mask)

    mask_real, x_edges, y_edges = np.histogram2d(
        samps[0], samps[1], bins=[bins, bins], weights=samps[2], normed=False
    )
    mask_imag, x_edges, y_edges = np.histogram2d(
        samps[0], samps[1], bins=[bins, bins], weights=samps[3], normed=False
    )

    mask_real = np.rot90(mask_real)
    mask_imag = np.rot90(mask_imag)

    mask_real /= mask
    mask_imag /= mask

    gridded_vis = np.zeros((2, N, N))
    gridded_vis[0] = mask_real
    gridded_vis[1] = mask_imag
    return gridded_vis


def convert_amp_phase(data, sky_sim=False):
    if sky_sim:
        amp = np.abs(data)
        phase = np.angle(data)
        amp = (np.log10(amp + 1e-10) / 10) + 1
        data = np.stack((amp, phase), axis=1)
    else:
        data[:, 0] = np.sqrt(data[:, 0] ** 2 + data[:, 1] ** 2) / 1e3
        data[:, 1] = np.angle(data[:, 0] + 1j * data[:, 1])
        data[:, 0] = (np.log10(data[:, 0] + 1e-10) / 10) + 1
    return data


if __name__ == "__main__":
    create_gridded_data_set(
        "/net/big-tank/POOL/projects/radio/test_rime/create_dataset.toml"
    )
