import torch

# Registry of telescope-specific parameters for elevation-dependent noise.
# Add new telescopes here to extend support.
# Elevation knots in degrees; temperature contributions in K.
_TELESCOPES = {
    # Digitized from https://skaafrica.atlassian.net/wiki/
    # spaces/ESDKB/pages/277315585/MeerKAT+specifications (Figure 6)
    # at 1284 MHz
    "meerkat": {
        "dish_diameter": 13.5,  # metres
        "el_knots": torch.tensor(
            [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
        ),
        "t_atm": torch.tensor(
            [5.05, 3.80, 3.10, 2.65, 2.35, 2.15, 1.85, 1.65, 1.52, 1.43, 1.38]
        ),
        "t_spill_h": torch.tensor(
            [2.58, 2.52, 2.48, 2.45, 2.43, 2.42, 2.43, 2.50, 2.65, 2.88, 3.15]
        ),
        "t_spill_v": torch.tensor(
            [2.60, 2.54, 2.50, 2.47, 2.45, 2.44, 2.46, 2.55, 2.70, 2.92, 3.18]
        ),
    },
    # VLBA at 15 GHz (2 cm / X-band, 15.363 GHz)
    # Reference: VLBA Observational Status Summary 2026B,
    #   https://science.nrao.edu/facilities/vlba/docs/manuals/oss/bands-perf
    #   Typical zenith SEFD = 543 Jy, Peak Gain = 0.111 K/Jy
    #   → geometric aperture efficiency η ≈ 0.62, Tsys(zenith) ≈ 60 K
    # T_atm: computed from the airmass model
    #   T_atm(el) = T_phys * (1 - exp(-τ_z / sin(el)))
    #   with T_phys = 265 K and τ_z = 0.05 (representative zenith opacity
    #   at 15 GHz under median conditions).
    # T_spill: approximate values for a 25 m alt-az cassegrain at 15 GHz;
    #   no published H/V separation available, so t_spill_h == t_spill_v.
    # For noise_mode='tsys', set noise_level ≈ Tsys(55°)/η ≈ 100 K.
    "vlba": {
        "dish_diameter": 25.0,  # metres
        "el_knots": torch.tensor(
            [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
        ),
        "t_atm": torch.tensor(
            [
                46.55,
                36.06,
                29.56,
                25.22,
                22.12,
                19.83,
                16.74,
                14.87,
                13.73,
                13.12,
                12.92,
            ]
        ),
        "t_spill_h": torch.tensor(
            [3.50, 3.20, 3.00, 2.90, 2.80, 2.70, 2.70, 2.75, 2.85, 2.95, 3.05]
        ),
        "t_spill_v": torch.tensor(
            [3.50, 3.20, 3.00, 2.90, 2.80, 2.70, 2.70, 2.75, 2.85, 2.95, 3.05]
        ),
    },
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
    el_deg: torch.Tensor, telescope: str = "meerkat", pol: str = "mean"
) -> torch.Tensor:
    """Returns T_spill + T_atm as a function of elevation for a given telescope.

    Parameters
    ----------
    el_deg : torch.Tensor
        Elevation in degrees, any shape.
    telescope : str
        Telescope name. Must be a key in the telescope registry.
    pol : str
        'H', 'V', or 'mean' (default).

    Returns
    -------
    torch.Tensor
        Elevation-dependent Tsys contribution in K, same shape as el_deg.
    """
    spec = _TELESCOPES[telescope]
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
) -> torch.Tensor:
    """Per-baseline SEFD derived from elevation and T_sys/η.

    Parameters
    ----------
    el1_deg, el2_deg : torch.Tensor
        Per-baseline elevation in degrees for each antenna. Shape: [n_baselines].
    tsys_over_eta_ref : float
        T_sys/η at reference elevation (55°) in K.
    telescope : str
        Telescope name. Must be a key in the telescope registry.

    Returns
    -------
    torch.Tensor
        Geometric-mean SEFD per baseline in Jy. Shape: [n_baselines].
    """
    k_B = 1.38e-23
    d = _TELESCOPES[telescope]["dish_diameter"]
    A_geom = torch.pi * (d / 2) ** 2

    ref_el = torch.tensor(55.0, device=el1_deg.device)
    t_ref = elevation_tsys_contribution(ref_el, telescope=telescope)

    delta1 = elevation_tsys_contribution(el1_deg, telescope=telescope) - t_ref
    delta2 = elevation_tsys_contribution(el2_deg, telescope=telescope) - t_ref

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
        Telescope name for elevation-dependent corrections.
        Must be a key in the telescope registry. Only used for ``mode='tsys'``.

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
        if telescope not in _TELESCOPES:
            raise ValueError(
                f"Unknown telescope '{telescope}'. "
                f"Available: {list(_TELESCOPES.keys())}"
            )
        sefd = sefd_from_elevation(el1_deg, el2_deg, noise_value, telescope=telescope)
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
