import numpy as np

from pyvisgen.utils.logging import setup_logger

LOGGER = setup_logger()


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
