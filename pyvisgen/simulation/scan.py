from math import pi

import torch
from astropy.constants import c
from torch import nn
from torch.special import bessel_j1


class RIME(nn.Module):
    def __init__(
        self,
        bas,
        lm,
        rd,
        ra,
        dec,
        ant_diam,
        spw_low,
        spw_high,
        corrupted=False,
    ):
        """Calculates visibilities using RIME

        Parameters
        ----------
        img: torch.tensor
            sky distribution
        bas : dataclass object
            baselines dataclass
        lm : 2d array
            lm grid for FOV
        spw_low : float
            lower wavelength
        spw_high : float
            higher wavelength

        Returns
        -------
        2d tensor
            Returns visibility for every baseline
        """
        super().__init__()
        self.fourier_kernel = fourier_kernel(bas, lm, spw_low, spw_high)

        self.corrupted = corrupted
        if self.corrupted:
            self.beam = calc_beam(rd, ra, dec, ant_diam, spw_low, spw_high)

    def forward(self, img):
        with torch.no_grad():
            K1, K2 = self.fourier_kernel
            X1, X2 = img * K1, img * K2
            del K1, K2
            if self.corrupted:
                E1, E2 = self.beam
                X1 = E1[..., None] * X1 * E1[..., None]
                del E1
                X2 = E2[..., None] * X2 * E2[..., None]
                del E2
            vis = integrate(X1, X2)
        return vis


@torch.compile
def fourier_kernel(bas, lm, spw_low, spw_high):
    """Calculates Fouriertransformation Kernel for every baseline and pixel in lm grid.

    Parameters
    ----------
    bas : dataclass object
        baseline information
    lm : 2d array
        lm grid for FOV
    spw_low : float
        lower wavelength
    spw_high : float
        higher wavelength

    Returns
    -------
    3d tensor
        Return Fourier Kernel for every pixel in lm grid and given baselines.
        Shape is given by lm axes and baseline axis
    """
    u_cmplt = torch.cat((bas[0], bas[1]))
    v_cmplt = torch.cat((bas[3], bas[4]))
    w_cmplt = torch.cat((bas[6], bas[7]))

    l = lm[..., 0]
    m = lm[..., 1]
    n = torch.sqrt(1 - l**2 - m**2)

    ul = u_cmplt[..., None] * l
    vm = v_cmplt[..., None] * m
    wn = w_cmplt[..., None] * (n - 1)
    del l, m, n, u_cmplt, v_cmplt, w_cmplt

    K1 = torch.exp(
        -2
        * pi
        * 1j
        * (ul / c.value * spw_low + vm / c.value * spw_low + wn / c.value * spw_low)
    )[..., None, None]
    K2 = torch.exp(
        -2
        * pi
        * 1j
        * (ul / c.value * spw_high + vm / c.value * spw_high + wn / c.value * spw_high)
    )[..., None, None]
    del ul, vm, wn
    return K1, K2


@torch.compile
def calc_beam(rd, ra, dec, ant_diam, spw_low, spw_high):
    diameters = ant_diam.to(rd.device)
    theta = angularDistance(rd, ra, dec)
    tds = diameters * theta[..., None]

    E1 = jinc(2 * pi / c.value * spw_low * tds)
    E2 = jinc(2 * pi / c.value * spw_high * tds)

    assert E1.shape == E2.shape
    return E1, E2


@torch.compile
def angularDistance(rd, ra, dec):
    """Calculates angular distance from source position

    Parameters
    ----------
    rd : 3d tensor
        every pixel containing ra and dec
    ra : float
        right ascension of source position
    dec : float
        declination of source position

    Returns
    -------
    2d array
        Returns angular Distance for every pixel in rd grid with respect
        to source position
    """
    r = rd[..., 0] - torch.deg2rad(ra.to(rd.device))
    d = rd[..., 1] - torch.deg2rad(dec.to(rd.device))
    theta = torch.arcsin(torch.sqrt(r**2 + d**2))
    return theta


@torch.compile
def jinc(x):
    """Create jinc function.

    Parameters
    ----------
    x : array
        value of (?)

    Returns
    -------
    array
        value of jinc function at x
    """
    jinc = torch.ones(x.shape, device=x.device).double()
    jinc[x != 0] = 2 * bessel_j1(x[x != 0]) / x[x != 0]
    return jinc


@torch.compile
def integrate(X1, X2):
    """Summation over (l,m) and avering over time and freq

    Parameters
    ----------
    X1 : 3d tensor
        visibility for every (l,m) and baseline for freq1
    X2 : 3d tensor
        visibility for every (l,m) and baseline for freq2

    Returns
    -------
    2d tensor
    Returns visibility for every baseline
    """
    X_f = torch.stack((X1, X2))
    int_m = torch.sum(X_f, dim=2)
    del X_f
    # only integrate for 1 sky dimension
    # 2d sky is reshaped to 1d by sensitivity mask
    # int_l = torch.sum(int_m, dim=2)
    # del int_m
    int_f = 0.5 * torch.sum(int_m, dim=0)
    del int_m
    X_t = torch.stack(torch.split(int_f, int(int_f.shape[0] / 2), dim=0))
    del int_f
    int_t = 0.5 * torch.sum(X_t, dim=0)
    del X_t
    return int_t
