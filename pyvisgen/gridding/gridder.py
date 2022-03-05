import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pyvisgen.fits.data import fits_data


def create_gridded_dataset(fits_data_path):
    fits_files = fits_data(fits_data_path)
    size = len(fits_files)

    for i in tqdm(range(1)):
        freq_data, base_freq = fits_files.get_freq_data(i)
        uv_data = fits_files.get_uv_data(i)

        gridded_data = grid_data(uv_data, base_freq)


def grid_data(uv_data, base_freq):
    freq = base_freq

    cmplx = uv_data["DATA"]
    real = np.squeeze(cmplx[..., 0, 0])
    imag = np.squeeze(cmplx[..., 0, 1])
    # weight = np.squeeze(cmplx[..., 0, 2])
    ap = np.sqrt(real ** 2 + imag ** 2)
    ph = np.angle(real + 1j * imag)
    samps = np.array(
        [
            np.append(uv_data["UU--"] * freq, -uv_data["UU--"] * freq),
            np.append(uv_data["VV--"] * freq, -uv_data["VV--"] * freq),
            np.append(ap, ap),
            np.append(ph, -ph),
        ]
    )

    # Generate Mask
    u_0 = samps[0]
    v_0 = samps[1]
    N = 256  # image size, needs to be from config file
    mask = np.zeros((N, N, u_0.shape[0]))
    fov = (
        0.0256 * np.pi / (3600 * 180)
    )  # hard code #default 0.00018382, FoV from VLBA 163.7
    print(fov)
    delta_u = 1 / (fov)
    print(delta_u)
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
    gridded_vis[0] = (np.log10(gridded_vis[0] + 1e-10) / 10) + 1
    gridded_vis[1] = np.matmul(mask, samps[3].T) / points

    print(gridded_vis.shape)

    plt.imshow(gridded_vis[1])
    plt.colorbar()
    plt.show()

    return 1


if __name__ == "__main__":
    create_gridded_dataset("/net/big-tank/POOL/projects/radio/test_rime/build/uvfits")

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