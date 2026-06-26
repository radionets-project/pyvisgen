import sysconfig
import tomllib
import warnings
from functools import cache
from pathlib import Path

import torch


def _noise_config_dir() -> Path:
    root = sysconfig.get_path("data", sysconfig.get_default_scheme())
    return Path(root) / "share" / "resources" / "noise_configs"


def available_telescopes() -> list[str]:
    """Return the names of all telescopes with installed noise configs."""
    return sorted(p.stem for p in _noise_config_dir().glob("*.toml"))


@cache
def _load_telescope_toml(telescope: str) -> dict:
    path = _noise_config_dir() / f"{telescope}.toml"
    if not path.exists():
        avail = available_telescopes()
        warnings.warn(
            f"No noise config found for telescope '{telescope}'. Available: {avail}",
            stacklevel=4,
        )
        raise ValueError(f"Unknown telescope '{telescope}'. Available: {avail}")
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def _get_band_spec(telescope: str, band: str | None = None) -> dict:
    """Load the noise parameter dict for *telescope* and optional *band*.

    If *band* is None the first band defined in the config file is used.
    The returned dict has keys: dish_diameter, el_knots, t_atm,
    t_spill_h, t_spill_v (all as torch.Tensor except dish_diameter).
    """
    cfg = _load_telescope_toml(telescope)
    bands = cfg.get("bands", {})
    if not bands:
        raise ValueError(f"Noise config for '{telescope}' defines no bands.")

    if band is None:
        band = next(iter(bands))
    elif band not in bands:
        raise ValueError(
            f"Band '{band}' not found for telescope '{telescope}'. "
            f"Available: {list(bands.keys())}"
        )

    bspec = bands[band]
    return {
        "dish_diameter": cfg["dish_diameter"],
        "el_knots": torch.tensor(bspec["el_knots"]),
        "t_atm": torch.tensor(bspec["t_atm"]),
        "t_spill_h": torch.tensor(bspec["t_spill_h"]),
        "t_spill_v": torch.tensor(bspec["t_spill_v"]),
    }


def _interp1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Linear interpolation, equivalent to np.interp, works on batched x."""
    xp = xp.to(x.device)
    fp = fp.to(x.device)
    x = x.clamp(xp[0], xp[-1])
    idx = torch.searchsorted(xp, x) - 1
    idx = idx.clamp(0, len(xp) - 2)
    t = (x - xp[idx]) / (xp[idx + 1] - xp[idx])
    return fp[idx] + t * (fp[idx + 1] - fp[idx])


def elevation_tsys_contribution(
    el_deg: torch.Tensor,
    telescope: str = "meerkat",
    band: str | None = None,
    pol: str = "mean",
) -> torch.Tensor:
    """Returns T_spill + T_atm as a function of elevation for a given telescope.

    Parameters
    ----------
    el_deg : torch.Tensor
        Elevation in degrees, any shape.
    telescope : str
        Telescope name. Must match a file in the noise_configs resource directory.
    band : str, optional
        Frequency band name defined in the telescope config. Defaults to the
        first band in the config file.
    pol : str
        'H', 'V', or 'mean' (default).

    Returns
    -------
    torch.Tensor
        Elevation-dependent Tsys contribution in K, same shape as el_deg.
    """
    spec = _get_band_spec(telescope, band)
    t_atm = _interp1d(el_deg, spec["el_knots"], spec["t_atm"])

    if pol == "H":
        t_spill = _interp1d(el_deg, spec["el_knots"], spec["t_spill_h"])
    elif pol == "V":
        t_spill = _interp1d(el_deg, spec["el_knots"], spec["t_spill_v"])
    else:
        t_spill = _interp1d(
            el_deg, spec["el_knots"], (spec["t_spill_h"] + spec["t_spill_v"]) / 2
        )

    return t_atm + t_spill


def sefd_from_elevation(
    el1_deg: torch.Tensor,
    el2_deg: torch.Tensor,
    tsys_over_eta_ref: float,
    telescope: str = "meerkat",
    band: str | None = None,
) -> torch.Tensor:
    """Per-baseline SEFD derived from elevation and T_sys/η.

    Parameters
    ----------
    el1_deg, el2_deg : torch.Tensor
        Per-baseline elevation in degrees for each antenna. Shape: [n_baselines].
    tsys_over_eta_ref : float
        T_sys/η at reference elevation (55°) in K.
    telescope : str
        Telescope name. Must match a file in the noise_configs resource directory.
    band : str, optional
        Frequency band name. Defaults to the first band in the config file.

    Returns
    -------
    torch.Tensor
        Geometric-mean SEFD per baseline in Jy. Shape: [n_baselines].
    """
    k_B = 1.38e-23
    spec = _get_band_spec(telescope, band)
    d = spec["dish_diameter"]
    A_geom = torch.pi * (d / 2) ** 2

    ref_el = torch.tensor(55.0, device=el1_deg.device)
    t_ref = elevation_tsys_contribution(ref_el, telescope=telescope, band=band)

    delta1 = (
        elevation_tsys_contribution(el1_deg, telescope=telescope, band=band) - t_ref
    )
    delta2 = (
        elevation_tsys_contribution(el2_deg, telescope=telescope, band=band) - t_ref
    )

    sefd1 = 2 * k_B * (tsys_over_eta_ref + delta1) / A_geom * 1e26
    sefd2 = 2 * k_B * (tsys_over_eta_ref + delta2) / A_geom * 1e26
    return torch.sqrt(sefd1 * sefd2)


def compute_noise_std(obs, sefd: torch.Tensor) -> torch.Tensor:
    """Convert SEFD to noise standard deviation via the radiometer equation.

    Parameters
    ----------
    obs : Observation
        Observation object providing bandwidth and integration time.
    sefd : torch.Tensor
        SEFD in Jy. Scalar or shape [n_baselines].

    Returns
    -------
    torch.Tensor
        Noise std per baseline.
    """
    eta = 0.93
    chan_width = obs.bandwidths[0]
    exposure = obs.int_time
    return (1 / eta) * sefd / torch.sqrt(2 * exposure * chan_width)


def generate_noise(
    shape: tuple,
    obs,
    noise_value: float,
    mode: str = "sefd",
    el1_deg: torch.Tensor | None = None,
    el2_deg: torch.Tensor | None = None,
    telescope: str = "meerkat",
    band: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate complex Gaussian visibility noise and natural weights.

    Parameters
    ----------
    shape : tuple
        Shape of the visibility array (n_baselines, n_channels, ...).
    obs : Observation
        Observation object.
    noise_value : float
        SEFD in Jy when ``mode='sefd'``, or T_sys/η in K when ``mode='tsys'``.
    mode : str
        ``'sefd'``: uniform SEFD noise, no elevation dependence (backward compatible).
        ``'tsys'``: elevation-dependent noise derived from system temperature.
    el1_deg, el2_deg : torch.Tensor, optional
        Per-baseline elevation in degrees. Required when ``mode='tsys'``.
    telescope : str
        Telescope name for elevation-dependent corrections. Must match a file in
        the noise_configs resource directory. Only used for ``mode='tsys'``.
    band : str, optional
        Frequency band name defined in the telescope config. Defaults to the
        first band in the config file. Only used for ``mode='tsys'``.

    Returns
    -------
    noise : torch.Tensor
        Complex noise array of the given shape.
    weights : torch.Tensor
        Natural weights (1/σ²) of shape (n_baselines, n_channels).
    """
    if mode == "tsys":
        if el1_deg is None or el2_deg is None:
            raise ValueError("el1_deg and el2_deg are required for mode='tsys'")
        sefd = sefd_from_elevation(
            el1_deg, el2_deg, noise_value, telescope=telescope, band=band
        )
    elif mode == "sefd":
        device = el1_deg.device if el1_deg is not None else torch.device("cpu")
        sefd = torch.full(
            (shape[0],), float(noise_value), dtype=torch.float64, device=device
        )
    else:
        raise ValueError(f"Unknown noise mode '{mode}'. Choose 'sefd' or 'tsys'.")

    std = compute_noise_std(obs, sefd)
    weights = (1.0 / std**2).unsqueeze(-1).expand(shape[0], shape[1])
    std = std.reshape(-1, *([1] * (len(shape) - 1)))
    std_expanded = std.expand(shape)
    noise = torch.normal(mean=0.0, std=std_expanded)
    noise = noise + 1.0j * torch.normal(mean=0.0, std=std_expanded)
    return noise, weights
