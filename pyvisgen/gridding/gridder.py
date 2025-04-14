import astropy.constants as const
import numpy as np
import torch
from numpy.exceptions import AxisError

from pyvisgen.gridding.alt_gridder import ms2dirty_python_fast


def ducc0_gridding(uv_data, freq_data):
    vis_ = uv_data["DATA"]
    vis = np.array([vis_[:, 0, 0, 0, 0, 0, 0] + 1j * vis_[:, 0, 0, 0, 0, 0, 1]]).T
    vis_compl = np.array([vis_[:, 0, 0, 0, 0, 0, 0] + 1j * vis_[:, 0, 0, 0, 0, 0, 1]]).T
    uu = np.array(uv_data["UU--"], dtype=np.float64)
    uu_compl = np.array(-uv_data["UU--"], dtype=np.float64)
    vv = np.array(uv_data["VV--"], dtype=np.float64)
    vv_compl = np.array(-uv_data["VV--"], dtype=np.float64)
    ww = np.array(uv_data["WW--"], dtype=np.float64)
    ww_compl = np.array(uv_data["WW--"], dtype=np.float64)
    uvw = np.stack([uu, vv, ww]).T
    uvw_compl = np.stack([uu_compl, vv_compl, ww_compl]).T
    uvw *= const.c.value
    uvw_compl *= const.c.value
    # complex conjugated
    uvw = np.append(uvw, uvw_compl, axis=0)
    vis = np.append(vis, vis_compl)

    freq = freq_data[1]
    freq = (freq_data[0]["IF FREQ"] + freq).reshape(-1, 1)[0]
    wgt = np.ones((vis.shape[0], 1))
    mask = None

    wgt[vis == 0] = 0
    if mask is None:
        mask = np.ones(wgt.shape, dtype=np.uint8)
    mask[wgt == 0] = False

    DEG2RAD = np.pi / 180
    # nthreads = 4
    epsilon = 1e-4
    # do_wgridding = False
    # verbosity = 1

    # do_sycl = False  # True
    # do_cng = False  # True

    # ntries = 1

    fov_deg = 0.02  # 1e-5  # 3.3477833333331884e-5

    npixdirty = 64  # get_npixdirty(uvw, freq, fov_deg, mask)
    pixsize = fov_deg / npixdirty * DEG2RAD

    # mintime = 1e300

    grid = ms2dirty_python_fast(
        uvw, freq, vis, npixdirty, npixdirty, pixsize, pixsize, epsilon, False
    )
    grid = np.rot90(np.fft.fftshift(grid))
    # assert grid.shape[0] == 256
    return grid


def grid_data(uv_data, freq_data, conf):
    cmplx = uv_data["DATA"]
    # real and imag for Stokes I, linear feed
    real = np.squeeze(cmplx[..., 0, 0, 0]) + np.squeeze(cmplx[..., 0, 3, 0])
    imag = np.squeeze(cmplx[..., 0, 0, 1]) + np.squeeze(cmplx[..., 0, 3, 1])

    # visibility weighting not yet implemented
    # weight = np.squeeze(cmplx[..., 0, 2])

    freq = freq_data[1]
    IF_bands = (freq_data[0]["IF FREQ"] + freq).reshape(-1, 1)

    u = np.repeat([uv_data["UU--"]], real.shape[1], axis=0)
    v = np.repeat([uv_data["VV--"]], real.shape[1], axis=0)
    u = (u * IF_bands).T.ravel()
    v = (v * IF_bands).T.ravel()

    real = real.ravel()
    imag = imag.ravel()

    samps = np.array(
        [
            np.append(-u, u),
            np.append(-v, v),
            np.append(real, real),
            np.append(-imag, imag),
        ]
    )
    # Generate Mask
    N = conf["grid_size"]  # image size
    fov = conf["grid_fov"] * np.pi / (3600 * 180)

    delta = 1 / fov

    # bins are shifted by delta/2 so that maximum in uv space matches maximum
    # in numpy fft
    bins = np.arange(start=-((N + 1) / 2) * delta, stop=(N / 2) * delta, step=delta)

    mask, *_ = np.histogram2d(samps[0], samps[1], bins=[bins, bins], density=False)
    mask[mask == 0] = 1

    mask_real, x_edges, y_edges = np.histogram2d(
        samps[0], samps[1], bins=[bins, bins], weights=samps[2], density=False
    )
    mask_imag, x_edges, y_edges = np.histogram2d(
        samps[0], samps[1], bins=[bins, bins], weights=samps[3], density=False
    )

    mask_real /= mask
    mask_imag /= mask

    assert mask_real.shape == (conf["grid_size"], conf["grid_size"])
    gridded_vis = np.zeros((2, N, N))
    gridded_vis[0] = mask_real
    gridded_vis[1] = mask_imag

    return gridded_vis


def grid_vis_loop_data(
    uu: torch.tensor,
    vv: torch.tensor,
    vis_data,
    freq_bands: list,
    conf: dict,
    stokes_comp: int = "I",
    polarization: str | None = None,
) -> np.array:
    """Grid data returned by :func:`~pyvisgen.simulation.vis_loop`.

    Parameters
    ----------
    uu : :func:`~torch.tensor`
    vv : :func:`~torch.tensor`
    vis_data : :class:`~pyvisgen.simulation.Visibilities`
        :class:`~pyvisgen.simulation.Visibilities` dataclass object
        containing the visibilities measured by the array.
    freq_bands : list
        List of frequency bands of the observation.
    conf : dict
        Dictionary containing the configuration of the observation.
    stokes_comp : int, optional
        Index of the stokes component to grid. Defaults to stokes I.
        Default: 0
    polarization: str or None, optional
        Polarization type in the data. Default: ``'I'``

    Returns
    -------
    gridded_vis : :func:`~np.array`
        Array of gridded visibilities.
    """
    if vis_data.ndim != 7:
        if vis_data.ndim == 3:
            vis_data = np.stack(
                [vis_data.real, vis_data.imag, np.ones(vis_data.shape)],
                axis=3,
            )[:, None, None, :, None, ...]
    else:
        raise ValueError("Expected vis_data to be of dimension 3 or 7")

    if isinstance(freq_bands, float):
        freq_bands = [freq_bands]

    uu /= const.c
    vv /= const.c

    u = np.array([uu * np.array(freq) for freq in freq_bands]).ravel()
    v = np.array([vv * np.array(freq) for freq in freq_bands]).ravel()

    # get stokes visibilities depending on stokes component to grid
    # and polarization mode
    stokes_vis = _get_stokes_vis(vis_data, stokes_comp, polarization)
    try:
        stokes_vis = stokes_vis.swapaxes(0, 1).ravel()
    except AxisError:
        stokes_vis = stokes_vis.ravel()

    real = stokes_vis.real
    imag = stokes_vis.imag

    samps = np.array(
        [
            np.concatenate([-u, u]),
            np.concatenate([-v, v]),
            np.concatenate([real, real]),
            np.concatenate([-imag, imag]),
        ]
    )

    # Generate Mask
    N = conf["grid_size"]  # image size
    fov = np.deg2rad(conf["grid_fov"] / 3600)
    delta = 1 / fov

    # bins are shifted by delta/2 so that maximum in uv space matches maximum
    # in numpy fft
    bins = np.arange(
        start=-((N + 1) / 2) * delta,
        stop=((N + 1) / 2) * delta,
        step=delta,
        dtype=np.float128,
    )

    mask, *_ = np.histogram2d(samps[0], samps[1], bins=[bins, bins], density=False)
    mask[mask == 0] = 1

    mask_real, x_edges, y_edges = np.histogram2d(
        samps[0], samps[1], bins=[bins, bins], weights=samps[2], density=False
    )
    mask_imag, x_edges, y_edges = np.histogram2d(
        samps[0], samps[1], bins=[bins, bins], weights=samps[3], density=False
    )

    mask_real /= mask
    mask_imag /= mask

    if mask_real.shape != (conf["grid_size"], conf["grid_size"]):
        raise ValueError(
            "shape mismatch: Expected mask_real to be "
            f"of shape {(conf['grid_size'], conf['grid_size'])}"
        )

    gridded_vis = np.zeros((2, N, N))
    gridded_vis[0] = mask_real
    gridded_vis[1] = mask_imag

    return gridded_vis


def compute_single_stokes_component(
    vis_data, stokes_comp_1: int, stokes_comp_2: int, sign: str
):
    """Computes single stokes components I, Q, U, or V from visibility
    data for gridding.

    Parameters
    ----------
    vis_data : :class:`~pyvisgen.simulation.Visibilities`
        :class:`~pyvisgen.simulation.Visibilities` dataclass object
        containing the visibilities measured by the array.
    stokes_comp_1 : int
        Index of first stokes visibility.
    stokes_comp_2 : int
        Index of second stokes visibility.
    sign : str
        Wether to add or substract ``stokes_comp_1`` and ``stokes_comp_2``.
        Valid values are ``'+'`` or ``'-'``.
    """
    if sign not in "+-":
        raise ValueError("'sign' can only be '+' or '-'!")
    match sign:
        case "+":
            real = vis_data[..., stokes_comp_1, 0] + vis_data[..., stokes_comp_2, 0]
            imag = vis_data[..., stokes_comp_1, 1] + vis_data[..., stokes_comp_2, 1]
        case "-":
            real = vis_data[..., stokes_comp_1, 0] - vis_data[..., stokes_comp_2, 0]
            imag = vis_data[..., stokes_comp_1, 1] - vis_data[..., stokes_comp_2, 1]

    vis = real + 1j * imag

    return vis


def _get_stokes_vis(vis_data, stokes_comp: str, polarization: str):
    """Returns the stokes visibility for a given stokes component
    depending on polarization.
    """
    if polarization == "circular":
        match stokes_comp:
            case "I":
                stokes_vis = compute_single_stokes_component(vis_data, 0, 3, "+")
            case "Q":
                stokes_vis = compute_single_stokes_component(vis_data, 1, 2, "+")
            case "U":
                stokes_vis = compute_single_stokes_component(vis_data, 1, 2, "-")
            case "V":
                stokes_vis = compute_single_stokes_component(vis_data, 0, 3, "-")
            case "I+V":
                stokes_vis = vis_data[..., 0, 0] + 1j * vis_data[..., 0, 1]
            case "Q+U":
                stokes_vis = vis_data[..., 1, 0] + 1j * vis_data[..., 1, 1]
            case "Q-U":
                stokes_vis = vis_data[..., 2, 0] + 1j * vis_data[..., 2, 1]
            case "I-V":
                stokes_vis = vis_data[..., 3, 0] + 1j * vis_data[..., 3, 1]
    else:
        match stokes_comp:
            case "I":
                stokes_vis = compute_single_stokes_component(vis_data, 0, 3, "+")
            case "Q":
                stokes_vis = compute_single_stokes_component(vis_data, 0, 3, "-")
            case "U":
                stokes_vis = compute_single_stokes_component(vis_data, 1, 2, "+")
            case "V":
                stokes_vis = compute_single_stokes_component(vis_data, 1, 2, "-")
            case "I+Q":
                stokes_vis = vis_data[..., 0, 0] + 1j * vis_data[..., 0, 1]
            case "U+V":
                stokes_vis = vis_data[..., 1, 0] + 1j * vis_data[..., 1, 1]
            case "U-V":
                stokes_vis = vis_data[..., 2, 0] + 1j * vis_data[..., 2, 1]
            case "I-Q":
                stokes_vis = vis_data[..., 3, 0] + 1j * vis_data[..., 3, 1]

    return np.squeeze(stokes_vis)
