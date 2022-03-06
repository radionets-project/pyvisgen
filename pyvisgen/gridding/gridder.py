import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from pyvisgen.fits.data import fits_data
from pyvisgen.utils.config import read_data_set_conf


def create_gridded_data_set(config):
    conf = read_data_set_conf(config)
    out_path_fits = Path(conf["out_path"])
    out_path = out_path_fits.parent / "gridded_data/"
    ######
    out_path = Path(
        "/net/big-tank/POOL/projects/radio/test_rime/build_test/gridded_data"
    )
    out_path_fits = Path(
        "/net/big-tank/POOL/projects/radio/test_rime/build_test/uvfits"
    )
    ######
    print(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    fits_files = fits_data(out_path_fits)
    size = len(fits_files)
    num_bundles = int(size // conf["bundle_size"])
    if conf["num_test_images"] > 0:
        budles_test = int(conf["num_test_images"] // conf["bundle_size"])
        size -= conf["num_test_images"]
    else:
        bundle_test = 0

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
        print(gridded_data_train.shape)
        print(gridded_data_train[0].shape)


def grid_data(uv_data, conf):
    freq = conf["base_freq"]

    cmplx = uv_data["DATA"]
    real = np.squeeze(cmplx[..., 0, 0])
    imag = np.squeeze(cmplx[..., 0, 1])
    # weight = np.squeeze(cmplx[..., 0, 2])
    # ap = np.sqrt(real ** 2 + imag ** 2)
    # ph = np.angle(real + 1j * imag)
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
    )  # hard code #default 0.00018382, FoV from VLBA 163.7 <- wrong! depends on setting of simulations
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
    # gridded_vis[0] = (np.log10(gridded_vis[0] + 1e-10) / 10) + 1
    gridded_vis[1] = np.matmul(mask, samps[3].T) / points

    # print(gridded_vis.shape)

    # plt.imshow(gridded_vis[1])
    # plt.colorbar()
    # plt.show()

    return gridded_vis


if __name__ == "__main__":
    create_gridded_data_set(
        "/net/big-tank/POOL/projects/radio/test_rime/create_dataset.toml"
    )

    # img = np.zeros((size, 256, 256))
    # samps = np.zeros((size, 4, 21000))  # hard code
    # for i in tqdm(range(size)):
    #     sampled = bundle_paths_input[i]
    #     target = bundle_paths_target[i]

    #     img[i] = np.asarray(Image.open(str(target)))
    #     # img[i] = img[i]/np.sum(img[i])

    # print(f"\n Gridding VLBI data set.\n")

    # # Generate Mask
    # u_0 = samps[0][0]
    # v_0 = samps[0][1]
    # N = 63  # hard code
    # mask = np.zeros((N, N, u_0.shape[0]))
    # fov = 0.00018382 * np.pi / (3600 * 180)  # hard code #default 0.00018382
    # # delta_u = 1/(fov*N/256) # hard code
    # delta_u = 1 / (fov)
    # for i in range(N):
    #     for j in range(N):
    #         u_cell = (j - N / 2) * delta_u
    #         v_cell = (i - N / 2) * delta_u
    #         mask[i, j] = ((u_cell <= u_0) & (u_0 <= u_cell + delta_u)) & (
    #             (v_cell <= v_0) & (v_0 <= v_cell + delta_u)
    #         )

    # mask = np.flip(mask, [0])
    # points = np.sum(mask, 2)
    # points[points == 0] = 1
    # samp_img = np.zeros((size, 2, N, N))
    # img_resized = np.zeros((size, N, N))
    # for i in tqdm(range(samps.shape[0])):
    #     samp_img[i][0] = np.matmul(mask, samps[i][2].T) / points
    #     samp_img[i][0] = (np.log10(samp_img[i][0] + 1e-10) / 10) + 1
    #     samp_img[i][1] = np.matmul(mask, samps[i][3].T) / points
    #     img_resized[i] = cv2.resize(img[i], (N, N))
    #     img_resized[i] = img_resized[i] / np.sum(img_resized[i])

    # # truth_fft = np.array([np.fft.fft2(np.fft.fftshift(img)) for im in img_resized])
    # truth_fft = np.fft.fftshift(
    #     np.fft.fft2(np.fft.fftshift(img_resized, axes=(1, 2)), axes=(1, 2)), axes=(1, 2)
    # )
    # fft_scaled_truth = prepare_fft_images(truth_fft, True, False)

    # out = data_path + "/samp_train0.h5"
    # save_fft_pair(out, samp_img[:2300], fft_scaled_truth[:2300])
    # out = data_path + "/samp_valid0.h5"
    # save_fft_pair(out, samp_img[2300:], fft_scaled_truth[2300:])