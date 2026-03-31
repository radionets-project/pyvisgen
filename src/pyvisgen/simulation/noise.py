import torch

# Digitized from https://skaafrica.atlassian.net/wiki/...
# .../spaces/ESDKB/pages/277315585/MeerKAT+specifications (Figure 6)
# at 1284 MHz
# Elevation in degrees
_EL_KNOTS = torch.tensor(
    [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
)

# Temperature contributions in K
_T_ATM = torch.tensor(
    [5.05, 3.80, 3.10, 2.65, 2.35, 2.15, 1.85, 1.65, 1.52, 1.43, 1.38]
)
_T_SPILL_H = torch.tensor(
    [2.58, 2.52, 2.48, 2.45, 2.43, 2.42, 2.43, 2.50, 2.65, 2.88, 3.15]
)
_T_SPILL_V = torch.tensor(
    [2.60, 2.54, 2.50, 2.47, 2.45, 2.44, 2.46, 2.55, 2.70, 2.92, 3.18]
)


def _interp1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Linear interpolation, equivalent to np.interp, works on batched x."""
    xp = xp.to(x.device)
    fp = fp.to(x.device)
    idx = torch.searchsorted(xp, x.clamp(xp[0], xp[-1])) - 1
    idx = idx.clamp(0, len(xp) - 2)
    t = (x - xp[idx]) / (xp[idx + 1] - xp[idx])
    return fp[idx] + t * (fp[idx + 1] - fp[idx])


def elevation_tsys_contribution(
    el_deg: torch.Tensor, pol: str = "mean"
) -> torch.Tensor:
    """
    Returns T_spill + T_atm as a function of elevation (degrees).

    Parameters
    ----------
    el_deg : torch.Tensor
        Elevation in degrees, any shape.
    pol : str
        'H', 'V', or 'mean' (default).

    Returns
    -------
    torch.Tensor
        Elevation-dependent Tsys contribution in K, same shape as el_deg.
    """
    t_atm = _interp1d(el_deg, _EL_KNOTS, _T_ATM)

    if pol == "H":
        t_spill = _interp1d(el_deg, _EL_KNOTS, _T_SPILL_H)
    elif pol == "V":
        t_spill = _interp1d(el_deg, _EL_KNOTS, _T_SPILL_V)
    else:
        t_spill = _interp1d(el_deg, _EL_KNOTS, (_T_SPILL_H + _T_SPILL_V) / 2)

    return t_atm + t_spill


def sefd_from_elevation(el1_deg, el2_deg, Tsys_over_eta_ref, T_rec=0.0):
    """Per-baseline SEFD from elevation pairs. Shape: [n_baselines]"""
    k_B = 1.38e-23
    A_geom = torch.pi * (13.5 / 2) ** 2

    delta1 = elevation_tsys_contribution(el1_deg) - elevation_tsys_contribution(
        torch.tensor(55.0, device=el1_deg.device)
    )
    delta2 = elevation_tsys_contribution(el2_deg) - elevation_tsys_contribution(
        torch.tensor(55.0, device=el2_deg.device)
    )
    sefd1 = 2 * k_B * (Tsys_over_eta_ref + delta1) / A_geom * 1e26
    sefd2 = 2 * k_B * (Tsys_over_eta_ref + delta2) / A_geom * 1e26
    return torch.sqrt(sefd1 * sefd2)  # geometric mean per baseline


def compute_noise_std(obs, SEFD):
    """SEFD: scalar or [n_baselines] tensor"""
    eta = 0.93
    chan_width = obs.bandwidths[0]
    exposure = obs.int_time
    return (1 / eta) * SEFD / torch.sqrt(2 * exposure * chan_width)


def generate_noise(shape, obs, el1_deg, el2_deg, Tsys_over_eta_ref):
    sefd = sefd_from_elevation(el1_deg, el2_deg, Tsys_over_eta_ref)
    std = compute_noise_std(obs, sefd)  # [n_baselines]
    weights = (
        (1.0 / std**2).unsqueeze(-1).expand(shape[0], shape[1])
    )  # [n_baselines, n_channels]
    std = std.reshape(-1, *([1] * (len(shape) - 1)))  # [n_baselines, 1, 1, 1]
    std_expanded = std.expand(shape)
    noise = torch.normal(mean=0.0, std=std_expanded)
    noise = noise + 1.0j * torch.normal(mean=0.0, std=std_expanded)
    return noise, weights
