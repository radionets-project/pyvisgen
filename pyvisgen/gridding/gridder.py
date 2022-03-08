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
    out_path = out_path_fits.parent / "gridded_data/"
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
    # num_bundles = int(size // conf["bundle_size"])
    # if conf["num_test_images"] > 0:
    #     bundle_test = int(conf["num_test_images"] // conf["bundle_size"])
    #     size -= conf["num_test_images"]
    # else:
    #     bundle_test = 0

    size_valid = conf["train_valid_split"] * size
    size_train = size - size_valid
    bundle_train = int(size_train // conf["bundle_size"])
    bundle_valid = int(size_valid // conf["bundle_size"])

    for i in tqdm(range(bundle_train)):
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
        truth_fft_train = np.fft.fftshift(
            np.fft.fft2(np.fft.fftshift(sky_dist_train, axes=(1, 2)), axes=(1, 2)),
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
        truth_fft_valid = np.fft.fftshift(
            np.fft.fft2(np.fft.fftshift(sky_dist_valid, axes=(1, 2)), axes=(1, 2)),
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
    u_0 = samps[0]
    v_0 = samps[1]
    N = conf["grid_size"]  # image size, needs to be from config file
    mask = np.zeros((N, N, u_0.shape[0]))
    fov = (
        conf["fov_size"] * np.pi / (3600 * 180)
    )  # hard code #default 0.00018382, FoV from VLBA 163.7 <- wrong!
    # depends on setting of simulations

    delta_u = 1 / (fov)
    for i in range(N):
        for j in range(N):
            u_cell = (j - N / 2) * delta_u
            v_cell = (i - N / 2) * delta_u
            mask[i, j] = ((u_cell <= u_0) & (u_0 <= u_cell + delta_u)) & (
                (v_cell <= v_0) & (v_0 <= v_cell + delta_u)
            )

    mask = np.flip(mask, [0])
    points = np.sum(mask, 2)
    points[points == 0] = 1
    gridded_vis = np.zeros((2, N, N))
    gridded_vis[0] = np.matmul(mask, samps[2].T) / points
    gridded_vis[1] = np.matmul(mask, samps[3].T) / points
    return gridded_vis


def convert_amp_phase(data, sky_sim=False):
    if sky_sim:
        amp = np.abs(data)
        phase = np.angle(data)
        amp = (np.log10(amp + 1e-10) / 10) + 1
        data = np.stack((amp, phase), axis=1)
    else:
        data[:, 0] = np.sqrt(data[:, 0] ** 2 + data[:, 1] ** 2)
        data[:, 1] = np.angle(data[:, 0] + 1j * data[:, 1])
        data[:, 0] = (np.log10(data[:, 0] + 1e-10) / 10) + 1
    return data


if __name__ == "__main__":
    create_gridded_data_set(
        "/net/big-tank/POOL/projects/radio/test_rime/create_dataset.toml"
    )
