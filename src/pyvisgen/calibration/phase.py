from typing import Literal

import numpy as np
import torch
from tqdm.auto import tqdm

from pyvisgen.simulation.observation import Observation
from pyvisgen.simulation.visibility import Visibilities
from pyvisgen.utils.logging import setup_logger

torch.set_default_dtype(torch.float64)
LOGGER = setup_logger(namespace=__name__)

__all__ = [
    "PhaseCalibration",
    "compute_phase_residuals",
]


class PhaseCalibration:
    """Phase calibration for radio interferometer visibilities.

    This class implements phase-only self-calibration using iterative
    least-squares minimization of the difference between observed and
    model visibilities.

    Parameters
    ----------
    method : str, optional
        Calibration method. Options are:
        - 'selfcal': Phase-only self-calibration
        Default: 'selfcal'
    max_iter : int, optional
        Maximum number of iterations.  Default: 50
    tolerance : float, optional
        Convergence tolerance for the phase solutions. Default: 1e-6
    ref_antenna : int, optional
        Reference antenna index.  If None, the first antenna is used.
        Default: None
    device : str, optional
        Torch device ('cuda' or 'cpu'). Default: 'cuda'
    show_progress : bool, optional
        If True, show progress bar during iterations. Default: False

    Attributes
    ----------
    gains : torch.tensor
        Complex gain solutions per antenna and frequency.
    converged : bool
        Whether the calibration converged.
    residuals : list
        List of residuals at each iteration.


    """

    def __init__(
        self,
        method: Literal["selfcal"] = "selfcal",
        max_iter: int = 50,
        tolerance: float = 1e-6,
        ref_antenna: int | None = None,
        device: str = "cuda",
        show_progress: bool = False,
    ) -> None:
        """Initialize phase calibration object."""
        self.method = method
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.ref_antenna = ref_antenna
        self.device = torch.device(device)
        self.show_progress = show_progress

        self.gains = None
        self.converged = False
        self.residuals = []

    def selfcal(
        self,
        V_obs: torch.tensor,
        sky_model: torch.tensor,
        gains: torch.tensor,
        baseline_nums: torch.tensor,
        n_ant: int,
        ref_ant: int,
        obs: Observation,
        **vis_loop_kwargs,
    ) -> torch.tensor:
        """Solve using phase-only self-calibration (Stefcal algorithm).

        This implements a simplified version of the Stefcal algorithm for
        phase-only calibration.

        Parameters
        ----------
        V_obs : torch.tensor
            Observed visibilities, shape (n_vis, n_freq).
        sky_model : torch.tensor
            Sky model visibilities, shape (n_vis, n_freq).
        gains : np.ndarray
            Initial gain estimates, shape (n_ant, n_freq).
        baseline_nums : torch.tensor
            Baseline numbers encoding antenna pairs.
        n_ant : int
            Number of antennas.
        ref_ant : int
            Reference antenna index.
        obs : Observation
            Observation object containing array information.
        vis_loop_kwargs : dict
            Additional keyword arguments for vis_loop when computing model visibilities.

        Returns
        -------
        gains : torch.tensor
            Solved phase-only gains.
        """

        #  baseline numbers to antenna indices
        ant1 = ((baseline_nums / 256).long() - 1).cpu().numpy()
        ant2 = ((baseline_nums % 256).long() - 1).cpu().numpy()

        V_obs = (V_obs.V_11 + V_obs.V_22) / 2.0
        n_freq = V_obs.shape[1]

        iterator = range(self.max_iter)
        if self.show_progress:
            iterator = tqdm(iterator, desc="Phase calibration iterations")

        # Set default parameters for vis_loop if not provided
        vis_loop_params = {
            "num_threads": vis_loop_kwargs.get("num_threads", 10),
            "noisy": False,
            "mode": vis_loop_kwargs.get("mode", "full"),
            "batch_size": vis_loop_kwargs.get("batch_size", "auto"),
            "show_progress": vis_loop_kwargs.get("show_progress", False),
            "normalize": vis_loop_kwargs.get("normalize", True),
            "ft": vis_loop_kwargs.get("ft", "default"),
            "atmospheric_effects": None,
            "tec_values": None,
            "include_faraday": False,
        }

        from pyvisgen.simulation.visibility import vis_loop

        # Generate model visibilities
        vis_model_obj = vis_loop(obs=obs, SI=sky_model, **vis_loop_params)

        # Extract model visibilities (use Stokes I: (V_11 + V_22) / 2)
        V_model = ((vis_model_obj.V_11 + vis_model_obj.V_22) / 2.0).to(self.device)

        # Ensure V_model is 2D
        if V_model.ndim == 1:
            V_model = V_model.unsqueeze(1)

        if V_model.shape != V_obs.shape:
            LOGGER.error(
                f"Shape mismatch: V_obs={V_obs.shape}, V_model={V_model.shape}"
            )
        LOGGER.info(
            f"shape V_obs: {V_obs.shape}, shape V_model: {V_model.shape}, shape gains: {gains.shape}"
        )

        for iteration in iterator:
            gains_old = gains.clone()

            # Iterate over each antenna
            for ant in range(n_ant):
                if ant == ref_ant:
                    # Keep reference antenna phase at zero
                    gains[ant, :] = 1.0
                    continue

                # Find baselines involving this antenna
                mask1 = ant1 == ant
                mask2 = ant2 == ant

                if (not mask1.any()) and (not mask2.any()):
                    continue

                # Initialize accumulators for this antenna
                numerator = torch.zeros(n_freq, dtype=torch.cdouble, device=self.device)
                denominator = torch.zeros(
                    n_freq, dtype=torch.float64, device=self.device
                )

                # Baselines where this antenna is the first
                if mask1.any():
                    indices1 = np.where(mask1)[0]
                    for idx in indices1:
                        ant_j = ant2[idx]

                        V_o = V_obs[idx]
                        V_m = V_model[idx]
                        g_j = gains[ant_j, :]

                        numerator += V_o * torch.conj(V_m) * torch.conj(g_j)
                        denominator += torch.abs(V_m) ** 2 * torch.abs(g_j) ** 2

                # Baselines where this antenna is the second
                if mask2.any():
                    indices2 = np.where(mask2)[0]
                    for idx in indices2:
                        ant_i = ant1[idx]

                        V_o = V_obs[idx]
                        V_m = V_model[idx]
                        g_i = gains[ant_i, :]

                        numerator += torch.conj(V_o) * V_m * g_i
                        denominator += torch.abs(V_m) ** 2 * torch.abs(g_i) ** 2

                    # Update gain (phase-only: normalize to unit amplitude)
                mask_valid = denominator.abs() > 1e-12
                g_new = torch.ones_like(numerator)
                g_new[mask_valid] = numerator[mask_valid] / denominator[mask_valid]

                # Phase-only: set amplitude to 1, avoid divide-by-zero
                amp = torch.abs(g_new)
                amp = torch.clamp(amp, min=1e-12)
                gains[ant, :] = g_new / amp

            # Compute residual
            residual = torch.mean(torch.abs(gains - gains_old)).item()
            self.residuals.append(residual)

            if residual < self.tolerance:
                self.converged = True
                LOGGER.info(f"Convergence achieved at iteration {iteration + 1}")
                break
            else:
                LOGGER.debug(
                    f"Iteration {iteration + 1}: residual={residual:.2e}, max change in phase={torch.max(torch.angle(gains / gains_old)).item():.2f} radians"
                )

        return gains

    def apply(
        self,
        vis: Visibilities,
        gains: torch.Tensor | None = None,
    ) -> Visibilities:
        """Apply phase corrections to visibilities.

        Parameters
        ----------
        vis : Visibilities
            Visibilities dataclass object to correct.
        gains : torch.tensor, optional
            Complex gain solutions. If None, uses self.gains.  Default: None

        Returns
        -------
        vis_corrected : Visibilities
            Corrected visibilities dataclass object.

        Raises
        ------
        ValueError
            If gains have not been computed yet.
        """

        if gains is None:
            if self.gains is None:
                raise ValueError(
                    "No gains available. Run selfcal() first or provide gains."
                )
            gains = self.gains

        LOGGER.info("Applying phase corrections to visibilities")

        return self.apply_phase_corrections(vis=vis, gains=gains)

    def apply_phase_corrections(
        self,
        vis: Visibilities,
        gains: torch.tensor,
    ) -> Visibilities:
        """Apply antenna-based phase corrections to visibilities.

        Parameters
        ----------
        vis : Visibilities
            Input visibilities dataclass object.
        gains : torch.tensor
            Complex gain solutions of shape (n_ant, n_freq).

        Returns
        -------
        vis_corrected : Visibilities
            Phase-corrected visibilities.

        Notes
        -----
        The corrected visibility for baseline (i,j) is:


            V_{ij}^{\\text{corr}} = \\frac{V_{ij}^{\\text{obs}}}{g_i g_j^*}

        where g_i and g_j are the complex gains for
        antennas i and j.
        """
        baseline_nums = vis.base_num

        # Decode baseline numbers
        ant1 = (baseline_nums / 256).long() - 1
        ant2 = (baseline_nums % 256).long() - 1

        n_vis = len(baseline_nums)
        n_freq = vis.V_11.shape[1] if vis.V_11.ndim > 1 else 1

        # Get gains for each baseline
        g1 = gains[ant1, :]  # Shape: (n_vis, n_freq)
        g2 = gains[ant2, :]

        # Apply corrections: V_corr = V_obs / (g1 * conj(g2))
        correction = g1 * torch.conj(g2)

        vis.V_11 = vis.V_11 / correction
        vis.V_22 = vis.V_22 / correction
        vis.V_12 = vis.V_12 / correction
        vis.V_21 = vis.V_21 / correction

        return vis


def compute_phase_residuals(
    vis_obs: Visibilities,
    vis_model: Visibilities,
    gains: torch.Tensor,
) -> dict:
    """Compute phase residuals between observed and model visibilities
    after applying gains.
    Returns:
        dict: A dictionary containing the RMS, mean, and standard deviation of the phase
        residuals.
    """

    cal = PhaseCalibration()

    vis_obs_corr = Visibilities(
        vis_obs.V_11.clone(),
        vis_obs.V_22.clone(),
        vis_obs.V_12.clone(),
        vis_obs.V_21.clone(),
        vis_obs.num.clone(),
        vis_obs.base_num.clone(),
        vis_obs.u.clone(),
        vis_obs.v.clone(),
        vis_obs.w.clone(),
        vis_obs.date.clone(),
        vis_obs.linear_dop.clone(),
        vis_obs.circular_dop.clone(),
    )

    vis_obs_corr = cal.apply_phase_corrections(vis_obs_corr, gains)

    V_obs = (vis_obs_corr.V_11 + vis_obs_corr.V_22) / 2.0
    V_mod = (vis_model.V_11 + vis_model.V_22) / 2.0

    phase_diff = torch.angle(V_obs) - torch.angle(V_mod)
    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))

    return {
        "rms": torch.sqrt(torch.mean(phase_diff**2)).item(),
        "mean": torch.mean(phase_diff).item(),
        "std": torch.std(phase_diff).item(),
        "per_baseline_freq": phase_diff.cpu().numpy(),
    }
