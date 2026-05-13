import numpy as np


def calc_truth_fft(image):
    """Calculates the Fourier transform (image space -> uv-space)
    for a single image or a batch of images.

    This is shape independent as long as the last two axes are
    height and width, i.e. ``(..., H, W)``.

    Parameters
    ----------
    image : array_like, shape (..., H, W)
        (True) sky distribution.

    Returns
    -------
    fft_image : array_like, shape (..., H, W)
        Complex type array of the fft of the input image.
    """
    fft_image = np.fft.fftshift(
        np.fft.fft2(np.fft.fftshift(image, axes=(-2, -1)), axes=(-2, -1)),
        axes=(-2, -1),
    )
    return fft_image


def convert_amp_phase(data, sky_sim=False):
    if sky_sim:
        amp = np.abs(data)
        phase = np.angle(data)
        data = np.concatenate((amp, phase), axis=-3)
    else:
        test = data[:, 0] + 1j * data[:, 1]
        amp = np.abs(test)
        phase = np.angle(test)
        data = np.stack((amp, phase), axis=-3)

    return data


def convert_real_imag(data, sky_sim=False):
    if sky_sim:
        real = data.real
        imag = data.imag

        data = np.concatenate((real, imag), axis=-3)
    else:
        real = data[:, 0]
        imag = data[:, 1]

        data = np.stack((real, imag), axis=-3)

    return data
