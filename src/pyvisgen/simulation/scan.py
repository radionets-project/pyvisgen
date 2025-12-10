from __future__ import annotations

from math import pi
from typing import TYPE_CHECKING

import torch
from radioft.finufft import CupyFinufft
from scipy.constants import c
from torch.special import bessel_j1

if TYPE_CHECKING:
    from typing import Literal

    from numpy.typing import ArrayLike

torch.set_default_dtype(torch.float64)


finufft = CupyFinufft(image_size=512, fov_arcsec=1024, eps=1e-8)

__all__ = [
    "rime",
    "apply_finufft",
    "calc_fourier",
    "calc_feed_rotation",
    "calc_beam",
    "angular_distance",
    "jinc",
    "integrate",
]


# @torch.compile
def rime(
    img: ArrayLike,
    bas: ArrayLike,
    lm: ArrayLike,
    rd: ArrayLike,
    ra: ArrayLike,
    dec: ArrayLike,
    ant_diam: ArrayLike,
    spw_low: ArrayLike,
    spw_high: ArrayLike,
    polarization: str,
    mode: str,
    corrupted: bool = False,
    ft: Literal["default", "finufft", "reversed"] = "default",
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
    mode : str
        Select one of `'full'`, `'grid'`, or `'dense'` to get
        all valid baselines, a grid of unique baselines, or
        dense baselines.
    corrupted : bool, optional
        If ``True``, apply beam smearing to the simulated data.
        Default: ``False``
    ft : str, optional
        Sets the type of fourier transform used in the RIME.
        Choose one of ``'default'``, ``'finufft'`` (Flatiron Institute
        Nonuniform Fast Fourier Transform) or `'reversed'`.
        Default: ``'default'``

    Returns
    -------
    2d tensor
        Returns visibility for every baseline
    """
    if ft == "standard":
        with torch.no_grad():
            X1 = img.clone()
            X2 = img.clone()
            X1, X2 = calc_fourier(X1, X2, bas, lm, spw_low, spw_high)

            if polarization and mode != "dense":
                X1, X2 = calc_feed_rotation(X1, X2, bas, polarization)

            if corrupted:
                X1, X2 = calc_beam(X1, X2, rd, ra, dec, ant_diam, spw_low, spw_high)

            vis = integrate(X1, X2)
    if ft == "reversed":
        with torch.no_grad():
            img = torch.repeat_interleave(img.clone()[None], len(bas[2]), dim=0)
            X1 = img.clone()
            X2 = img.clone()
            if polarization and mode != "dense":
                X1, X2 = calc_feed_rotation(X1, X2, bas, polarization)

            if corrupted:
                X1, X2 = calc_beam(X1, X2, rd, ra, dec, ant_diam, spw_low, spw_high)

            X1, X2 = calc_fourier(X1, X2, bas, lm, spw_low, spw_high)
            vis = integrate(X1, X2)
    if ft == "finufft":
        with torch.no_grad():
            X1 = img.clone()
            X2 = img.clone()
            # if polarization and mode != "dense":
            #     X1, X2 = calc_feed_rotation(X1, X2, bas, polarization)

            if corrupted:
                X1, X2 = calc_beam(X1, X2, rd, ra, dec, ant_diam, spw_low, spw_high)

            vis = apply_finufft(X1, X2, bas, lm, spw_low, spw_high)
    return vis


# @torch.compile
def apply_finufft(
    X1: torch.tensor,
    X2: torch.tensor,
    bas,
    lm: torch.tensor,
    spw_low: float,
    spw_high: float,
) -> tuple[torch.tensor, torch.tensor]:
    if torch.cuda.is_available():
        l_coords = lm[..., 0]
        m_coords = lm[..., 1]
        n_coords = torch.sqrt(1 - l_coords**2 - m_coords**2)

        u_coords_low = bas[2] / c * spw_low
        v_coords_low = bas[5] / c * spw_low
        w_coords_low = bas[8] / c * spw_low

        u_coords_high = bas[2] / c * spw_high
        v_coords_high = bas[5] / c * spw_high
        w_coords_high = bas[8] / c * spw_high

        n_baselines = len(bas[2])

        # Pre-allocate output
        vis = torch.empty([n_baselines, 2, 2], dtype=torch.complex128, device=X1.device)

        # Reshape input
        X1_flat = X1.reshape(4, -1)
        X2_flat = X2.reshape(4, -1)

        # Create CUDA streams for parallel execution of the 4 Stokes params
        streams = [torch.cuda.Stream() for _ in range(4)]

        results_low = []
        results_high = []

        for i in range(4):
            with torch.cuda.stream(streams[i]):
                vis_low = finufft.nufft(
                    X1_flat[i],
                    l_coords,
                    m_coords,
                    n_coords,
                    u_coords_low,
                    v_coords_low,
                    w_coords_low,
                )
                vis_high = finufft.nufft(
                    X2_flat[i],
                    l_coords,
                    m_coords,
                    n_coords,
                    u_coords_high,
                    v_coords_high,
                    w_coords_high,
                )
                results_low.append(vis_low)
                results_high.append(vis_high)

        # Synchronize all streams
        torch.cuda.synchronize()

        # Stack and reshape
        vis_low_all = torch.stack(results_low)
        vis_high_all = torch.stack(results_high)
        vis_avg = (vis_low_all + vis_high_all) / 2
        vis = vis_avg.T.reshape(n_baselines, 2, 2)
    else:
        raise RuntimeError(
            "CUDA is not available. Finufft backend requires a CUDA-enabled GPU to run."
        )

    return vis


# @torch.compile
def calc_fourier(
    X1: torch.tensor,
    X2: torch.tensor,
    bas,
    lm: torch.tensor,
    spw_low: float,
    spw_high: float,
) -> tuple[torch.tensor, torch.tensor]:
    """Calculates Fourier transformation kernel for
    every baseline and pixel in the lm grid.

    Parameters
    ----------
    X1 : :func:`~torch.tensor`
        Sky tensor.
    X2 : :func:`~torch.tensor`
        Sky tensor.
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
    u_valid = bas[2]
    v_valid = bas[5]
    w_valid = bas[8]

    l = lm[..., 0]  # noqa: E741
    m = lm[..., 1]
    n = torch.sqrt(1 - l**2 - m**2)

    ul = u_valid[..., None] * l
    vm = v_valid[..., None] * m
    wn = w_valid[..., None] * (n - 1)
    del l, m, n, u_valid, v_valid, w_valid

    K1 = torch.exp(-2 * pi * 1j * (ul + vm + wn) / c * spw_low)[..., None, None]
    K2 = torch.exp(-2 * pi * 1j * (ul + vm + wn) / c * spw_high)[..., None, None]
    del ul, vm, wn
    return X1 * K1, X2 * K2


# @torch.compile
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


# @torch.compile
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
