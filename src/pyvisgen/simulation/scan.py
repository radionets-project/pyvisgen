from math import pi

import torch
from scipy.constants import c
from torch.special import bessel_j1

torch.set_default_dtype(torch.float64)

try:
    from radioft.finufft import CupyFinufft

    _FINUFFT_AVAIL = True

except ImportError as e:
    _FINUFFT_AVAIL = False
    _FINUFFT_ERROR = str(e)


__all__ = [
    "RIMEScan",
    "apply_finufft",
    "calc_fourier",
    "calc_feed_rotation",
    "calc_beam",
    "angular_distance",
    "jinc",
    "integrate",
]


# class JonesMatrix(ABC):
#     @abstractmethod
#     def __call__(
#         self, X1: torch.Tensor, X2, torch: torch.Tensor, **ctx
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         """Apply the effect."""
#
#
# class FTKernel(JonesMatrix):
#     def __call__(self, X1, X2, bas=None, lm)


class RIMEScan:
    def __init__(self, ft, mode, obs, lm, rd, eps=1e-8):
        if _FINUFFT_AVAIL:
            self.cupy_finufft = CupyFinufft(
                image_size=obs.img_size, fov_arcsec=obs.fov, eps=eps
            )

        self.mode = mode
        self.ft = ft
        self.ft_func = getattr(self, ft)
        self.polarization = obs.polarization
        self.corrupted = obs.corrupted
        self.ra = obs.ra
        self.dec = obs.dec
        self.ant_diam = torch.unique(obs.array.diam)
        self.lm = lm
        self.rd = rd

    def __call__(
        self,
        img: torch.Tensor,
        bas,
        spw_low: torch.Tensor,
        spw_high: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            if self.ft == "reversed":
                img = torch.repeat_interleave(
                    img.clone()[None], len(bas.u_valid), dim=0
                )

            X1 = img.clone()
            X2 = img.clone()

            return self.ft_func(
                X1,
                X2,
                bas,
                spw_low,
                spw_high,
            )

    def default(
        self,
        X1: torch.Tensor,
        X2: torch.Tensor,
        bas: torch.Tensor,
        spw_low: torch.Tensor,
        spw_high: torch.Tensor,
    ) -> torch.Tensor:
        X1, X2 = calc_fourier(X1, X2, bas, self.lm, spw_low, spw_high)

        if self.polarization and self.mode != "dense":
            X1, X2 = calc_feed_rotation(X1, X2, bas, self.polarization)

        if self.corrupted:
            X1, X2 = calc_beam(
                X1,
                X2,
                self.rd,
                self.ra,
                self.dec,
                self.ant_diam,
                spw_low,
                spw_high,
            )

        vis = integrate(X1, X2)

        return vis

    def reversed(
        self,
        X1: torch.Tensor,
        X2: torch.Tensor,
        bas: torch.Tensor,
        spw_low: torch.Tensor,
        spw_high: torch.Tensor,
    ) -> torch.Tensor:
        if self.polarization and self.mode != "dense":
            X1, X2 = calc_feed_rotation(X1, X2, bas, self.polarization)

        if self.corrupted:
            X1, X2 = calc_beam(
                X1,
                X2,
                self.rd,
                self.ra,
                self.dec,
                self.ant_diam,
                spw_low,
                spw_high,
            )

        X1, X2 = calc_fourier(X1, X2, bas, self.lm, spw_low, spw_high)
        vis = integrate(X1, X2)

        return vis

    def finufft(
        self,
        X1: torch.Tensor,
        X2: torch.Tensor,
        bas: torch.Tensor,
        spw_low: torch.Tensor,
        spw_high: torch.Tensor,
    ) -> torch.Tensor:
        if not _FINUFFT_AVAIL:
            raise RuntimeError(_FINUFFT_ERROR)

        if self.corrupted:
            X1, X2 = calc_beam(
                X1,
                X2,
                self.rd,
                self.ra,
                self.dec,
                self.ant_diam,
                spw_low,
                spw_high,
            )

        vis = apply_finufft(
            X1, X2, bas, self.lm, spw_low, spw_high, finufft=self.cupy_finufft
        )

        return vis


def apply_finufft(
    X1: torch.Tensor,
    X2: torch.Tensor,
    bas,
    lm: torch.Tensor,
    spw_low: float | torch.Tensor,
    spw_high: float | torch.Tensor,
    finufft: CupyFinufft,
) -> torch.Tensor:  # pragma: no cover
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Finufft backend requires a CUDA-enabled GPU to run."
        )

    l_coords = lm[..., 0]
    m_coords = lm[..., 1]
    n_coords = torch.sqrt(1 - l_coords**2 - m_coords**2)

    u_coords_low = bas.u_valid / c * spw_low
    v_coords_low = bas.v_valid / c * spw_low
    w_coords_low = bas.w_valid / c * spw_low

    u_coords_high = bas.u_valid / c * spw_high
    v_coords_high = bas.v_valid / c * spw_high
    w_coords_high = bas.w_valid / c * spw_high

    n_baselines = len(bas.u_valid)

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

    return vis


def calc_fourier(
    X1: torch.Tensor,
    X2: torch.Tensor,
    bas,
    lm: torch.Tensor,
    spw_low: float | torch.Tensor,
    spw_high: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    u_valid = bas.u_valid
    v_valid = bas.v_valid
    w_valid = bas.w_valid

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


def calc_feed_rotation(
    X1: torch.Tensor,
    X2: torch.Tensor,
    bas,
    polarization: str,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    q1 = bas.q1_valid[..., None]
    q2 = bas.q2_valid[..., None]

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


def calc_beam(
    X1: torch.Tensor,
    X2: torch.Tensor,
    rd: torch.Tensor,
    ra: float | torch.Tensor,
    dec: float | torch.Tensor,
    ant_diam: torch.Tensor,
    spw_low: float | torch.Tensor,
    spw_high: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
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
def angular_distance(rd: torch.Tensor, ra: torch.Tensor, dec: torch.Tensor):
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
