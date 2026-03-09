"""Utility functions for calibration."""

import numpy as np
import torch

from pyvisgen.simulation.visibility import Visibilities
from pyvisgen.utils.logging import setup_logger

LOGGER = setup_logger(namespace=__name__)

__all__ = [
    "compute_closure_phases",
]


def compute_closure_phases(
    vis: Visibilities,
    triangles: list[tuple[int, int, int]] | None = None,
) -> torch.tensor:
    """Compute closure phases for antenna triangles.

    Closure phases are immune to antenna-based phase errors and are
    useful for assessing data quality.

    Parameters
    ----------
    vis : Visibilities
        Visibilities dataclass object.
    triangles : list of tuples, optional
        List of antenna triangles as (ant1, ant2, ant3).
        If None, all possible triangles are computed. Default: None

    Returns
    -------
    closure_phases : torch.tensor
        Closure phases in radians for each triangle.

    Notes
    -----
    The closure phase for triangle (i,j,k) is:

    .. math::

        \\phi_{ijk} = \\arg(V_{ij} V_{jk} V_{ki})

    References
    ----------
    Jennison (1958), "A phase sensitive interferometer technique for
    the measurement of the Fourier transforms of spatial brightness
    distributions of small angular extent"
    """
    LOGGER.info("Computing closure phases")

    baseline_nums = vis.base_num
    ant1 = ((baseline_nums / 256).long() - 1).numpy()
    ant2 = ((baseline_nums % 256).long() - 1).numpy()

    n_ant = max(ant1.max(), ant2.max()) + 1

    # Generate all triangles if not provided
    if triangles is None:
        triangles = []
        for i in range(n_ant):
            for j in range(i + 1, n_ant):
                for k in range(j + 1, n_ant):
                    triangles.append((i, j, k))

    closure_phases = []

    # Use Stokes I
    V = (vis.V_11 + vis.V_22) / 2.0

    for tri in triangles:
        i, j, k = tri

        # Find baselines
        idx_ij = np.where((ant1 == i) & (ant2 == j))[0]
        idx_jk = np.where((ant1 == j) & (ant2 == k))[0]
        idx_ki = np.where((ant1 == k) & (ant2 == i))[0]

        # Handle reversed baselines
        if len(idx_ij) == 0:
            idx_ij = np.where((ant1 == j) & (ant2 == i))[0]
            if len(idx_ij) > 0:
                V_ij = torch.conj(V[idx_ij[0]])
        else:
            V_ij = V[idx_ij[0]]

        if len(idx_jk) == 0:
            idx_jk = np.where((ant1 == k) & (ant2 == j))[0]
            if len(idx_jk) > 0:
                V_jk = torch.conj(V[idx_jk[0]])
        else:
            V_jk = V[idx_jk[0]]

        if len(idx_ki) == 0:
            idx_ki = np.where((ant1 == i) & (ant2 == k))[0]
            if len(idx_ki) > 0:
                V_ki = torch.conj(V[idx_ki[0]])
        else:
            V_ki = V[idx_ki[0]]

        # Compute closure phase
        closure_product = V_ij * V_jk * V_ki
        closure_phase = torch.angle(closure_product)
        closure_phases.append(closure_phase)

    closure_phases = torch.stack(closure_phases) if closure_phases else torch.tensor([])

    LOGGER.info(f"Computed {len(closure_phases)} closure phases")

    return closure_phases
