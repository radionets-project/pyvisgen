from math import pi

import torch
from scipy.constants import c
from torch.special import bessel_j1

torch.set_default_dtype(torch.float64)

__all__ = [
    "rime",
    "calc_fourier",
    "calc_feed_rotation",
    "calc_beam",
    "angular_distance",
    "jinc",
    "integrate",
]


@torch.compile
def rime(
    img,
    bas,
    lm,
    rd,
    ra,
    dec,
    ant_diam,
    spw_low,
    spw_high,
    polarization,
    mode,
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
    polarization : str
        Type of polarization.

    Returns
    -------
    2d tensor
        Returns visibility for every baseline
    """
    with torch.no_grad():
        X1, X2 = calc_fourier(img, bas, lm, spw_low, spw_high)

        if polarization and mode != "dense":
            X1, X2 = calc_feed_rotation(X1, X2, bas, polarization)

        if corrupted:
            X1, X2 = calc_beam(X1, X2, rd, ra, dec, ant_diam, spw_low, spw_high)

        vis = integrate(X1, X2)
    return vis


@torch.compile
def calc_fourier(
    img: torch.tensor,
    bas,
    lm: torch.tensor,
    spw_low: float,
    spw_high: float,
) -> tuple[torch.tensor, torch.tensor]:
    """Calculates Fourier transformation kernel for
    every baseline and pixel in the lm grid.

    Parameters
    ----------
    img : :func:`~torch.tensor`
        Sky distribution.
    bas : :class:`~pyvisgen.simulation.ValidBaselineSubset`
        :class:`~pyvisgen.simulation.Baselines` dataclass
        object containing information on u, v, and w coverage,
        and observation times.
    lm : :func:`~torch.tensor`
        lm grid for FOV.
    spw_low : float
        Lower wavelength.
    spw_high : float
        Higher wavelength.

    Returns
    -------
    tuple[torch.tensor, torch.tensor]
        Fourier kernels for every pixel in the lm grid and
        given baselines. Shape is given by lm axes and
        baseline axis.
    """
    # only use u, v, w valid
    u_cmplt = bas[2]
    v_cmplt = bas[5]
    w_cmplt = bas[8]

    l = lm[..., 0]  # noqa: E741
    m = lm[..., 1]
    n = torch.sqrt(1 - l**2 - m**2)

    ul = u_cmplt[..., None] * l
    vm = v_cmplt[..., None] * m
    wn = w_cmplt[..., None] * (n - 1)
    del l, m, n, u_cmplt, v_cmplt, w_cmplt

    K1 = torch.exp(-2 * pi * 1j * (ul + vm + wn) / c * spw_low)[..., None, None]
    K2 = torch.exp(-2 * pi * 1j * (ul + vm + wn) / c * spw_high)[..., None, None]
    del ul, vm, wn
    return img * K1, img * K2


@torch.compile
def calc_feed_rotation(
    X1: torch.tensor,
    X2: torch.tensor,
    bas,
    polarization: str,
) -> tuple[torch.tensor, torch.tensor]:
    """Calculates the feed rotation due to the parallactic
    angle rotation of the source over time.

    Parameters
    ----------
    X1 : :func:`~torch.tensor`
        Fourier kernel calculated via
        :func:`~pyvisgen.simulation.calc_fourier`.
    X2 : :func:`~torch.tensor`
        Fourier kernel calculated via
        :func:`~pyvisgen.simulation.calc_fourier`.
    bas : :class:`~pyvisgen.simulation.ValidBaselineSubset`
        :class:`~pyvisgen.simulation.Baselines` dataclass
        object containing information on u, v, and w coverage,
        observation times, and parallactic angles.
    polarization : str
        Type of polarization for the feed.

    Returns
    -------
    X1 : :func:`~torch.tensor`
        Fourier kernel with the applied feed rotation.
    X2 : :func:`~torch.tensor`
        Fourier kernel with the applied feed rotation.
    """
    q1 = bas[13][..., None]
    q2 = bas[16][..., None]

    xa = torch.zeros_like(X1)
    xb = torch.zeros_like(X2)

    if polarization == "circular":
        xa[..., 0, 0] = X1[..., 0, 0] * torch.exp(1j * q1)
        xa[..., 0, 1] = X1[..., 0, 1] * torch.exp(-1j * q1)
        xa[..., 1, 0] = X1[..., 1, 0] * torch.exp(1j * q1)
        xa[..., 1, 1] = X1[..., 1, 1] * torch.exp(-1j * q1)

        xb[..., 0, 0] = X2[..., 0, 0] * torch.exp(1j * q2)
        xb[..., 0, 1] = X2[..., 0, 1] * torch.exp(-1j * q2)
        xb[..., 1, 0] = X2[..., 1, 0] * torch.exp(1j * q2)
        xb[..., 1, 1] = X2[..., 1, 1] * torch.exp(-1j * q2)
    else:
        xa[..., 0, 0] = X1[..., 0, 0] * torch.cos(q1) - X1[..., 0, 1] * torch.sin(q1)
        xa[..., 0, 1] = X1[..., 0, 0] * torch.sin(q1) + X1[..., 0, 1] * torch.cos(q1)
        xa[..., 1, 0] = X1[..., 1, 0] * torch.cos(q1) - X1[..., 1, 1] * torch.sin(q1)
        xa[..., 1, 1] = X1[..., 1, 0] * torch.sin(q1) + X1[..., 1, 1] * torch.cos(q1)

        xb[..., 0, 0] = X2[..., 0, 0] * torch.cos(q2) - X2[..., 0, 1] * torch.sin(q2)
        xb[..., 0, 1] = X2[..., 0, 0] * torch.sin(q2) + X2[..., 0, 1] * torch.cos(q2)
        xb[..., 1, 0] = X2[..., 1, 0] * torch.cos(q2) - X2[..., 1, 1] * torch.sin(q2)
        xb[..., 1, 1] = X2[..., 1, 0] * torch.sin(q2) + X2[..., 1, 1] * torch.cos(q2)

    X1 = xa.detach().clone()
    X2 = xb.detach().clone()

    del xa, xb

    return X1, X2


@torch.compile
def calc_beam(
    X1: torch.tensor,
    X2: torch.tensor,
    rd: torch.tensor,
    ra: float,
    dec: float,
    ant_diam: torch.tensor,
    spw_low: float,
    spw_high: float,
) -> tuple[torch.tensor, torch.tensor]:
    """Computes the beam influence on the image.

    Parameters
    ----------
    X1 : :func:`~torch.tensor`
    X2 : :func:`~torch.tensor`
    rd : :func:`~torch.tensor`
    ra : float
    dec : float
    ant_diam : :func:`~torch.tensor`
    spw_low :  float
    spw_high :  float

    Returns
    -------
    tuple[torch.tensor, torch.tensor]
    """
    diameters = ant_diam.to(rd.device)
    theta = angular_distance(rd, ra, dec)
    tds = diameters * theta[..., None]

    E1 = jinc(2 * pi / c * spw_low * tds)
    E2 = jinc(2 * pi / c * spw_high * tds)

    assert E1.shape == E2.shape

    EXE1 = E1[..., None] * X1 * E1[..., None]
    del E1, X1

    EXE2 = E2[..., None] * X2 * E2[..., None]
    del E2, X2
    return EXE1, EXE2


@torch.compile
def angular_distance(rd, ra, dec):
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
    r = rd[..., 0]
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

    # sum over all sky pixels
    # only integrate for 1 sky dimension
    # 2d sky is reshaped to 1d by sensitivity mask
    int_lm = torch.sum(X_f, dim=2)
    del X_f

    # average two bandwidth edges
    int_f = 0.5 * torch.sum(int_lm, dim=0)
    del int_lm

    return int_f
