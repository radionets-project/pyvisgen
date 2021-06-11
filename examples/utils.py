import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from astropy.convolution import Gaussian2DKernel as kernel
from astropy.convolution import convolve
from astropy.wcs import WCS


def create_sky_sim(rc):
    wcs_input_dict = {
        "CTYPE1": "RA---SIN",
        "CUNIT1": "deg",
        "CDELT1": (rc["fov_size"] * np.pi / (3600 * 180) / 2) / 256,
        "CRPIX1": 1,
        "CRVAL1": rc["src_coord"].ra.value,
        "NAXIS1": 256,
        "CTYPE2": "DEC--SIN",
        "CUNIT2": "deg",
        "CDELT2": (rc["fov_size"] * np.pi / (3600 * 180) / 2) / 256,
        "CRPIX2": 1,
        "CRVAL2": rc["src_coord"].dec.value,
        "NAXIS2": 256,
        "equinox": 2000,
    }
    wcs_dict = WCS(wcs_input_dict)

    pos = np.random.randint(0, 256, size=(2, 10))
    flux = np.random.rand(10)

    grid = np.zeros((256, 256))
    grid[pos[0], pos[1]] = flux
    grid[[127, 128, 127, 128], [127, 128, 128, 127]] = 0.4

    k = kernel(2, 3)
    k_scale = k * (1 / k.array.max())

    beam = Ellipse(
        [10, 10],
        2 * 2.335,
        3 * 2.335,
        0,
        facecolor="white",
    )

    sky = convolve(grid, k_scale, normalize_kernel=False)

    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(projection=wcs_dict)
    im = ax.imshow(sky, cmap="inferno", origin="lower", interpolation="None")
    ax.add_artist(beam)
    ax.set_xlabel("right ascension")
    ax.set_ylabel("declination / deg")

    fig.colorbar(im, label="surface brightness / (Jy / beam)")
    fig.tight_layout()
    plt.show()

    return sky
