from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import scipy.ndimage
import torch
from tqdm.auto import tqdm
import numpy as np
from scipy import constants as const
from astropy import constants as astro_const
from astropy import units as un
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, AltAz

from typing import Optional


import torch.nn.functional as F
from pyvisgen.simulation.noise import generate_noise
from pyvisgen.simulation.scan import RIMEScan
from pyvisgen.utils.batch_size import adaptive_batch_size
from pyvisgen.utils.logging import setup_logger
from pyvisgen.calibration.phase import PhaseCalibration


if TYPE_CHECKING:
    from typing import Literal

torch.set_default_dtype(torch.float64)
LOGGER = setup_logger(namespace=__name__)

__all__ = [
    "Visibilities",
    "vis_loop",
    "Polarization",
    "generate_noise",
    "AtmosphericEffects",
]


@dataclass
class Visibilities:
    """Visibilities dataclass.

    Attributes
    ----------
    V_11 : :func:`~torch.tensor`
    V_22 : :func:`~torch.tensor`
    V_12 : :func:`~torch.tensor`
    V_21 : :func:`~torch.tensor`
    weights : :func:`~torch.tensor`
    num : :func:`~torch.tensor`
    base_num : :func:`~torch.tensor`
    u : :func:`~torch.tensor`
    v : :func:`~torch.tensor`
    w  : :func:`~torch.tensor`
    date : :func:`~torch.tensor`
    linear_dop : :func:`~torch.tensor`
    circular_dop : :func:`~torch.tensor`
    """

    V_11: torch.Tensor
    V_22: torch.Tensor
    V_12: torch.Tensor
    V_21: torch.Tensor
    weights: torch.Tensor
    num: torch.Tensor
    base_num: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    w: torch.Tensor
    date: torch.Tensor
    st_id_pairs: torch.Tensor
    linear_dop: torch.Tensor
    circular_dop: torch.Tensor

    def __getitem__(self, i):
        return Visibilities(*[getattr(self, f.name)[i] for f in fields(self)])

    def get_values(self):
        return torch.cat(
            [self.V_11[None], self.V_22[None], self.V_12[None], self.V_21[None]], dim=0
        ).permute(1, 2, 0)

    def add(self, visibilities):
        [
            setattr(
                self,
                f.name,
                torch.cat([getattr(self, f.name), getattr(visibilities, f.name)]),
            )
            for f in fields(self)
        ]


class Polarization:
    r"""Simulation of polarization.

    Creates the :math:`2\times 2` stokes matrix and simulates
    polarization if ``polarization`` is either ``'linear'``
    or ``'circular'``. Also computes the degree of polarization.

    Parameters
    ----------
    SI : :func:`~torch.tensor`
        Stokes I component, i.e. intensity distribution
        of the sky.
    sensitivity_cut : float
        Sensitivity cut, where only pixels above the value
        are kept.
    amp_ratio : float
        Sets the ratio of :math:`A_{X|R}`. The ratio of :math:`A_{Y|L}`
        is calculated as ``1 - amp_ratio``. If set to ``None``,
        a random value is drawn from a uniform distribution.
        See also: ``random_state``.
    delta : float
        Sets the phase difference of the amplitudes :math:`A_{X|R}`
        and :math:`A_{Y|L}`` of the sky distribution. Defines the
        measure of ellipticity.
    polarization : str
        Choose between ``'linear'`` or ``'circular'`` or ``None`` to
        simulate different types of polarizations or disable
        the simulation of polarization entirely.
    random_state : int
        Random state used when drawing ``amp_ratio`` and during
        the generation of the random polarization field.
    device : :class:`~torch.cuda.device`
        Torch device to select for computation.
    """

    def __init__(
        self,
        SI: torch.Tensor,
        sensitivity_cut: float,
        amp_ratio: float,
        delta: float | torch.Tensor,
        polarization: str,
        field_kwargs: dict,
        random_state: int,
        device: torch.device,
    ) -> None:
        """Creates the :math:`2\times 2` stokes matrix and simulates
        polarization if ``polarization`` is either ``'linear'``
        or ``'circular'``. Also computes the degree of polarization.

        Parameters
        ----------
        SI : :func:`~torch.tensor`
            Stokes I component, i.e. intensity distribution
            of the sky.
        sensitivity_cut : float
            Sensitivity cut, where only pixels above the value
            are kept.
        amp_ratio : float
            Sets the ratio of :math:`A_{X|R}`. The ratio of :math:`A_{Y|L}`
            is calculated as ``1 - amp_ratio``. If set to ``None``,
            a random value is drawn from a uniform distribution.
            See also: ``random_state``.
        delta : float
            Sets the phase difference of the amplitudes :math:`A_{X|R}`
            and :math:`A_{Y|L}`` of the sky distribution. Defines the
            measure of ellipticity.
        polarization : str
            Choose between ``'linear'`` or ``'circular'`` or ``None`` to
            simulate different types of polarizations or disable
            the simulation of polarization entirely.
        random_state : int
            Random state used when drawing ``amp_ratio`` and during
            the generation of the random polarization field.
        device : :class:`~torch.cuda.device`
            Torch device to select for computation.
        """
        self.sensitivity_cut = sensitivity_cut
        self.polarization = polarization
        self.device = device

        self.SI = SI.permute(dims=(1, 2, 0))

        if random_state:
            torch.manual_seed(random_state)

        if self.polarization and self.polarization in ["circular", "linear"]:
            self.polarization_field = self.rand_polarization_field(
                [self.SI.shape[0], self.SI.shape[1]],
                **field_kwargs,
            )

            if isinstance(delta, (float, int)):
                delta = torch.tensor(delta)

            self.delta = delta

            ax2 = amp_ratio if amp_ratio and amp_ratio >= 0 else torch.rand(1)

            if isinstance(ax2, torch.Tensor):
                ax2 = ax2.to(self.device)

            ay2 = 1 - ax2

            self.ax2 = self.SI[..., 0].detach().clone().to(self.device) * ax2
            self.ay2 = self.SI[..., 0].detach().clone().to(self.device) * ay2
        else:
            self.ax2 = self.SI[..., 0]
            self.ay2 = torch.zeros_like(self.ax2)

        self.I = torch.zeros(
            (self.SI.shape[0], self.SI.shape[1], 4), dtype=torch.cdouble
        )  # noqa: E741

    def linear(self) -> None:
        r"""Computes the stokes parameters I, Q, U, and V
        for linear polarization.

        This is done using the following equations:

        .. math::
            I &= A_X^2 + A_Y^2 \\
            Q &= A_X^2 - A_Y^2 \\
            U &= 2A_X A_Y \cos\delta_{XY} \\
            V &= -2A_X A_Y \sin\delta_{XY}
        """
        self.I[..., 0] = self.ax2 + self.ay2
        self.I[..., 1] = self.ax2 - self.ay2
        self.I[..., 2] = (
            2
            * torch.sqrt(self.ax2)
            * torch.sqrt(self.ay2)
            * torch.cos(torch.deg2rad(self.delta))
        )
        self.I[..., 3] = (
            -2
            * torch.sqrt(self.ax2)
            * torch.sqrt(self.ay2)
            * torch.sin(torch.deg2rad(self.delta))
        )

    def circular(self) -> None:
        r"""Computes the stokes parameters I, Q, U, and V
        for circular polarization.

        This is done using the following equations:

        .. math::

            I &= A_R^2 + A_L^2 \\
            Q &= 2A_R A_L \cos\delta_{RL} \\
            U &= -2A_R A_L \sin\delta_{RL} \\
            V &= A_R^2 - A_L^2
        """
        self.I[..., 0] = self.ax2 + self.ay2
        self.I[..., 1] = (
            2
            * torch.sqrt(self.ax2)
            * torch.sqrt(self.ay2)
            * torch.cos(torch.deg2rad(self.delta))
        )
        self.I[..., 2] = (
            -2
            * torch.sqrt(self.ax2)
            * torch.sqrt(self.ay2)
            * torch.sin(torch.deg2rad(self.delta))
        )
        self.I[..., 3] = self.ax2 - self.ay2

    def dop(self) -> None:
        """Computes the degree of polarization for each pixel."""
        mask = (self.ax2 + self.ay2) > 0

        # apply polarization_field to Q, U, and V only
        self.I[..., 1] *= self.polarization_field
        self.I[..., 2] *= self.polarization_field
        self.I[..., 3] *= self.polarization_field

        dop_I = self.I[..., 0].real.detach().clone()
        dop_I[~mask] = float("nan")
        dop_Q = self.I[..., 1].real.detach().clone()
        dop_Q[~mask] = float("nan")
        dop_U = self.I[..., 2].real.detach().clone()
        dop_U[~mask] = float("nan")
        dop_V = self.I[..., 3].real.detach().clone()
        dop_V[~mask] = float("nan")

        self.lin_dop = torch.sqrt(dop_Q**2 + dop_U**2) / dop_I
        self.circ_dop = torch.abs(dop_V) / dop_I

        del dop_I, dop_Q, dop_U, dop_V

    def stokes_matrix(self) -> tuple:
        """Computes and returns the 2 x 2 stokes matrix B.

        Returns
        -------
        B : torch.tensor
            2 x 2 stokes brightness matrix. Either for linear,
            circular or no polarization.
        mask : torch.tensor
            Mask of the sensitivity cut (Keep all px > sensitivity_cut).
        lin_dop : torch.tensor
            Degree of linear polarization of every pixel in the sky.
        circ_dop : torch.tensor
            Degree of circular polarization of every pixel in the sky.
        """
        # define 2 x 2 Stokes matrix
        B = torch.zeros(
            (self.SI.shape[0], self.SI.shape[1], 2, 2), dtype=torch.cdouble
        ).to(torch.device(self.device))

        if self.polarization == "linear":
            self.linear()
            self.dop()

            B[..., 0, 0] = self.I[..., 0] + self.I[..., 1]  # I + Q
            B[..., 0, 1] = self.I[..., 2] + 1j * self.I[..., 3]  # U + iV
            B[..., 1, 0] = self.I[..., 2] - 1j * self.I[..., 3]  # U - iV
            B[..., 1, 1] = self.I[..., 0] - self.I[..., 1]  # I - Q

        elif self.polarization == "circular":
            self.circular()
            self.dop()

            B[..., 0, 0] = self.I[..., 0] + self.I[..., 3]  # I + V
            B[..., 0, 1] = self.I[..., 1] + 1j * self.I[..., 2]  # Q + iU
            B[..., 1, 0] = self.I[..., 1] - 1j * self.I[..., 2]  # Q - iU
            B[..., 1, 1] = self.I[..., 0] - self.I[..., 3]  # I - V

        else:
            # No polarization applied
            self.I[..., 0] = self.SI[..., 0]
            self.polarization_field = torch.ones_like(self.I[..., 0])
            self.dop()

            B[..., 0, 0] = self.I[..., 0] + self.I[..., 1]  # I + Q
            B[..., 0, 1] = self.I[..., 2] + 1j * self.I[..., 3]  # U + iV
            B[..., 1, 0] = self.I[..., 2] - 1j * self.I[..., 3]  # U - iV
            B[..., 1, 1] = self.I[..., 0] - self.I[..., 1]  # I - Q

        # calculations only for px > sensitivity cut
        mask = (self.sensitivity_cut <= self.SI)[..., 0]
        B = B[mask]

        return B, mask, self.lin_dop, self.circ_dop

    def rand_polarization_field(
        self,
        shape: list[int] | int,
        order: list[int] | int = 1,
        random_state: int | None = None,
        scale: list | None = None,
        threshold: float | None = None,
    ) -> torch.Tensor:
        """
        Generates a random noise mask for polarization.

        Parameters
        ----------
        shape : array_like (M, N), or int
            The size of the sky image.
        order : array_like (M, N) or int, optional
            Morphology of the random noise. Higher values create
            more and smaller fluctuations. Default: ``1``.
        random_state : int, optional
            Random state for the random number generator. If ``None``,
            a random entropy is pulled from the OS. Default: ``None``.
        scale : array_like, optional
            Scaling of the distribution of the image. Default: ``[0, 1]``
        threshold : float, optional
            If not None, an upper threshold is applied to the image.
            Default: ``None``

        Returns
        -------
        im : torch.tensor
            An array containing random noise values between
            scale[0] and scale[1].
        """
        if random_state:
            torch.manual_seed(random_state)

        if isinstance(shape, int):
            shape = [shape]

        if not isinstance(shape, list):
            shape = list(shape)

        if len(shape) < 2:
            shape *= 2
        elif len(shape) > 2:
            raise ValueError("Expected len of 'shape' to be 2!")

        if isinstance(order, int | float):
            order = [order]

        if not isinstance(order, list):
            order = list(order)

        if len(order) < 2:
            order *= 2
        elif len(order) > 2:
            raise ValueError("Expected len of 'order' to be 2!")

        sigma = torch.mean(torch.tensor(shape).double()) / (40 * torch.tensor(order))

        im = torch.rand(shape)
        im = scipy.ndimage.gaussian_filter(im, sigma=sigma.numpy())

        if scale is None:
            scale = [im.min(), im.max()]

        if len(scale) != 2:
            raise ValueError("Expected len of 'scale' to be 2!")

        im_flatten = torch.from_numpy(im.flatten())
        im_argsort = torch.argsort(torch.argsort(im_flatten))
        im_linspace = torch.linspace(*scale, im_argsort.size()[0])
        uniform_flatten = im_linspace[im_argsort]

        im = torch.reshape(uniform_flatten, im.shape)

        if threshold:
            im = im[im < threshold]

        return im


def vis_loop(
    obs,
    SI: torch.Tensor,
    num_threads: int = 10,
    noise_level: float = 0,
    noise_mode: str = "sefd",
    telescope: str = "meerkat",
    band: str | None = None,
    mode: str = "full",
    batch_size: int | Literal["auto"] = "auto",
    show_progress: bool = False,
    normalize: bool = True,
    ft: Literal["default", "finufft", "reversed"] = "default",
    atmospheric_effects: Literal["ionosphere", "troposphere", "all"] | None = None,
    calibration: bool = False,
    calibration_tolerance: float = 1e-2,
) -> Visibilities:
    """Computes the visibilities of an observation.

    Parameters
    ----------
    obs :  Observation class object
        Observation class object generated by the
        `~pyvisgen.simulation.Observation` class.
    SI : torch.tensor
        Tensor containing the sky intensity distribution.
    num_threads : int, optional
        Number of threads used for intraoperative parallelism
        on the CPU. See `~torch.set_num_threads`. Default: 10
    noise_level : float, optional
        Noise amplitude: SEFD in Jy when ``noise_mode='sefd'``,
        or T_sys/η in K when ``noise_mode='tsys'``. Set to 0 to disable noise.
        Default: 0
    noise_mode : str, optional
        ``'sefd'``: uniform SEFD noise (backward compatible, no elevation dependence).
        ``'tsys'``: elevation-dependent noise from system temperature.
        Default: ``'sefd'``
    telescope : str, optional
        Telescope name for elevation-dependent Tsys corrections.
        Only used when ``noise_mode='tsys'``. Default: ``'meerkat'``
    mode : str, optional
        Select one of `'full'`, `'grid'`, or `'dense'` to get
        all valid baselines, a grid of unique baselines, or
        dense baselines.  Default: 'full'
    batch_size : int, optional
        Batch size for iteration over baselines. Default: 100
    show_progress : bool, optional
        If `True`, show a progress bar during the iteration over the
        batches of baselines. Default: False
    normalize : bool, optional
        If ``True``, normalize stokes matrix ``B`` by a factor 0.5.
        Default: ``True``
    ft : str, optional
        Sets the type of fourier transform used in the RIME.
        Choose one of ``'default'``, ``'finufft'`` (Flatiron Institute
        Nonuniform Fast Fourier Transform) or `'reversed'`.
        Default: ``'default'``
    atmospheric_effects : AtmosphericEffects, optional
        Whether and which atmospheric effects to include. Choose one of ``'ionosphere'``
        ``'troposphere'``, or ``'all'`` to include both.
        If ``None``, no atmospheric effects are applied. Default: ``None``
    calibration : bool, optional
        Whether to include phase self-calibration. Default: ``False``

    Returns
    -------
    visibilities : Visibilities
        Dataclass object containing visibilities and baselines.
        If atmospheric effects are applied, V_11, V_22, V_12, V_21
        have shape [n_baselines, n_freq, n_effects].
    """
    torch.set_num_threads(num_threads)
    torch._dynamo.config.suppress_errors = True

    if not (
        isinstance(batch_size, int)
        or (isinstance(batch_size, str) and batch_size == "auto")
    ):
        raise ValueError("Expected batch_size to be 'auto' or type int")

    pol = Polarization(
        SI,
        sensitivity_cut=obs.sensitivity_cut,
        polarization=obs.polarization,
        device=obs.device,
        field_kwargs=obs.field_kwargs,
        **obs.pol_kwargs,
    )

    B, mask, lin_dop, circ_dop = pol.stokes_matrix()

    lm = obs.lm[mask]
    rd = obs.rd[mask]

    # normalize visibilities to factor 0.5,
    # so that the Stokes I image is normalized to 1
    if normalize:
        B *= 0.5

    jones_atm = None
    atm_effects = None
    if atmospheric_effects is not None:
        times, time_axis_jd, time_list = AtmosphericEffects.timesteps(atm_effects, obs)

        atm_effects = AtmosphericEffects(
            obs,
            n_time=len(times),
            time_axis=times,
        )
        atm_effects.time_axis_jd = time_axis_jd

        if atmospheric_effects == "ionosphere":
            tec_values = atm_effects.slant_tec_from_iri(obs)
            LOGGER.info("Computing ionospheric Jones...")
            jones_atm = atm_effects.simulate_ionospheric_delay(
                tec_values,
            )
        elif atmospheric_effects == "troposphere":
            jones_atm = atm_effects.simulate_tropospheric_delay(obs=obs)
            LOGGER.info("Simulating tropospheric delay")
        elif atmospheric_effects == "all":
            tec_values = atm_effects.slant_tec_from_iri(obs)
            jones_iono = atm_effects.simulate_ionospheric_delay(
                tec_values,
            )
            jones_atm.append(jones_iono)

            jones_tropo = atm_effects.simulate_tropospheric_delay(obs=obs)
            jones_atm.append(jones_tropo)

        LOGGER.info(f"Jones shape: {jones_atm.shape}")
        for i, name in enumerate(atm_effects.effect_names):
            jones_i = jones_atm[i]
            LOGGER.info(f"\nEffect {i}: {name}")

            # Check if it's too close to identity
            identity_test = jones_i[:, :, 0, :, :] - torch.eye(
                2, device=jones_i.device, dtype=jones_i.dtype
            )
            LOGGER.info(
                f"  Distance from identity: {torch.abs(identity_test).mean():.6e}"
            )

            # Sample values
            LOGGER.info(f"  Sample J[0,0,0,:,:] = \n{jones_i[0, 0, 0, :, :]}")

    # calculate vis
    visibilities = Visibilities(
        torch.empty(size=[0] + [len(obs.waves_low)]),
        torch.empty(size=[0] + [len(obs.waves_low)]),
        torch.empty(size=[0] + [len(obs.waves_low)]),
        torch.empty(size=[0] + [len(obs.waves_low)]),
        torch.tensor([]),
        torch.tensor([]),
        torch.tensor([]),
        torch.tensor([]),
        torch.tensor([]),
        torch.tensor([]),
        torch.tensor([]),
        torch.empty(0, 2),
        torch.tensor([]),
        torch.tensor([]),
    )

    vis_num = torch.zeros(1)

    if mode == "full":
        bas = obs.baselines.get_valid_subset(obs.num_baselines, obs.device)
    elif mode == "grid":
        bas = obs.baselines.get_valid_subset(
            obs.num_baselines, obs.device
        ).get_unique_grid(obs.fov, obs.ref_frequency, obs.img_size, obs.device)
    elif mode == "dense":
        if obs.device == torch.device("cpu"):
            raise ValueError("Mode 'dense' is only available for GPU calculations!")

        # We cannot test this with our CI at the moment
        obs.calc_dense_baselines()  # pragma: no cover
        bas = obs.dense_baselines_gpu  # pragma: nocover
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if batch_size == "auto":
        batch_size = bas.baseline_nums.shape[0]

    visibilities = adaptive_batch_size(
        _batch_loop,
        batch_size,
        visibilities=visibilities,
        vis_num=vis_num,
        obs=obs,
        B=B,
        bas=bas,
        lm=lm,
        rd=rd,
        noise_level=noise_level,
        noise_mode=noise_mode,
        telescope=telescope,
        band=band,
        show_progress=show_progress,
        mode=mode,
        ft=ft,
        atmospheric_effects=atm_effects,
        jones_atm=jones_atm,
    )

    visibilities.linear_dop = lin_dop.to(obs.device)
    visibilities.circular_dop = circ_dop.to(obs.device)

    if calibration:
        LOGGER.info("Applying self-calibration")
        phase_cal = PhaseCalibration(
            obs,
            max_iter=8,
            tolerance=calibration_tolerance,
            ref_antenna=0,
            device=obs.device,
            show_progress=False,
        )

        n_ant = len(obs.array.st_num)
        n_freq = len(obs.waves_low)
        gains = torch.ones((n_ant, n_freq), dtype=torch.cdouble, device=obs.device)

        gains_solved = phase_cal.selfcal(
            V_obs=visibilities,
            sky_model=SI,
            gains=gains,
            baseline_nums=visibilities.base_num,
            n_ant=n_ant,
            ref_ant=1,
            obs=obs,
            num_threads=20,
            mode="full",
            batch_size=100,
            show_progress=False,
            normalize=True,
        )
        gains_solved = torch.as_tensor(gains_solved)
        visibilities = phase_cal.apply_phase_corrections(visibilities, gains_solved)

    return visibilities


def _batch_loop(
    batch_size: int,
    visibilities,
    vis_num: torch.Tensor,
    obs,
    B: torch.Tensor,
    bas,
    lm: torch.Tensor,
    rd: torch.Tensor,
    noise_level: float,
    noise_mode: str,
    telescope: str,
    band: str | None,
    show_progress: bool,
    mode: str,
    ft: Literal["default", "finufft", "reversed"] = "default",
    atmospheric_effects: Optional["AtmosphericEffects"] = None,
    jones_atm: Optional[torch.Tensor] = None,
):
    """Main simulation loop of pyvisgen. Computes visibilities
    batchwise.

    Parameters
    ----------
    batch_size : int
        Batch size for loop over Baselines dataclass object.
    visibilities : Visibilities
        Visibilities dataclass object.
    vis_num : torch.Tensor
        Number of visibilities.
    obs : Observation
        Observation class object.
    B : torch.tensor
        Stokes matrix containing stokes visibilities.
    bas : Baselines
        Baselines dataclass object.
    lm : torch.tensor
        lm grid.
    rd : torch.tensor
        rd grid.
    system_temp : float or bool
        Simulate noise based on system temperature with given value. If set to False,
        no noise is simulated.
    show_progress : bool
        If True, show a progress bar tracking the loop.
    mode : str
        Select one of `'full'`, `'grid'`, or `'dense'` to get
        all valid baselines, a grid of unique baselines, or
        dense baselines.
    ft : str, optional
        Sets the type of fourier transform used in the RIME.
        Choose one of ``'default'``, ``'finufft'`` (Flatiron Institute
        Nonuniform Fast Fourier Transform) or `'reversed'`.
        Default: ``'default'``

    Returns
    -------
    visibilities : Visibilities
        Visibilities dataclass object.
    """
    batches = torch.arange(bas.baseline_nums.shape[0]).split(batch_size)
    batches = tqdm(
        batches,
        position=0,
        disable=not show_progress,
        desc="Computing visibilities",
        postfix=f"Batch size: {batch_size}",
    )

    rime = RIMEScan(ft=ft, mode=mode, obs=obs, lm=lm, rd=rd)

    for p in batches:
        bas_p = bas[p]

        int_values = torch.cat(
            tensors=[
                rime(
                    B,
                    bas_p,
                    spw_low=wave_low,
                    spw_high=wave_high,
                )[None]
                for wave_low, wave_high in zip(obs.waves_low, obs.waves_high)
            ]
        )

        if int_values.numel() == 0:
            continue

        int_values = torch.swapaxes(int_values, 0, 1)

        # In case any row contains NaN
        int_values_nans = torch.isnan(int_values).any(dim=(1, 2, 3))
        int_values = int_values[~int_values_nans]

        # int_values shape: [n_baselines, n_freq, 2, 2]

        if atmospheric_effects is not None and jones_atm is not None:
            try:
                # LOGGER.info("Applying atmospheric effects to visibilities...")
                # Extract antenna indices from baseline numbers
                base_num = bas_p.baseline_nums[~int_values_nans].to(obs.device)
                n_ant = len(obs.array.st_num)

                # Decode baseline numbers to antenna indices
                # Standard VLBI encoding: base_num = (ant1+1) * 256 + (ant2+1)
                if n_ant <= 256:
                    st1 = (base_num // 256).long() - 1
                    st2 = (base_num % 256).long() - 1
                else:
                    st1 = (base_num // (n_ant + 1)).long() - 1
                    st2 = (base_num % (n_ant + 1)).long() - 1

                # Validate indices
                if (
                    (st1 < 0).any()
                    or (st1 >= n_ant).any()
                    or (st2 < 0).any()
                    or (st2 >= n_ant).any()
                ):
                    LOGGER.warning(
                        f"Invalid antenna indices: st1=[{st1.min()}, {st1.max()}], "
                        f"st2=[{st2.min()}, {st2.max()}], n_ant={n_ant}. Skipping atmospheric effects."
                    )
                    raise ValueError("Invalid antenna indices")

                # Combine all atmospheric effects via matrix multiplication
                # jones_atm shape: [n_effects, n_ant, n_freq, n_time, 2, 2]
                if jones_atm.shape[0] == 1:
                    # Only one effect
                    combined_jones = jones_atm[0]
                else:
                    # Multiple effects: combine via J_total = J_n @ ... @ J_2 @ J_1
                    combined_jones = jones_atm[0].clone()

                    for effect_idx in range(1, jones_atm.shape[0]):
                        # DEBUG: Check before combination
                        before_magnitude = torch.abs(combined_jones).mean()

                        combined_jones = torch.matmul(
                            jones_atm[effect_idx], combined_jones
                        )

                        # DEBUG: Check after combination
                        after_magnitude = torch.abs(combined_jones).mean()

                # DEBUG: Check combined Jones matrix
                # LOGGER.info(f"Combined Jones shape: {combined_jones.shape}")
                identity_test = combined_jones[:, :, 0, :, :] - torch.eye(
                    2, device=combined_jones.device, dtype=combined_jones.dtype
                )
                distance_from_identity = torch.abs(identity_test).mean()
                # LOGGER.info(
                #    f"Combined Jones distance from identity: {distance_from_identity:.6e}"
                # )

                obs_time_jd = bas_p.date[~int_values_nans].to(obs.device).double()
                time_axis_jd = atmospheric_effects.time_axis_jd.to(obs.device).double()

                time_idx = torch.argmin(
                    torch.abs(obs_time_jd[:, None] - time_axis_jd[None, :]), dim=1
                )
                # LOGGER.info("bas_p[10][:10] =", bas_p[10][:10])
                # LOGGER.info("unique time_idx =", torch.unique(time_idx)[:20])
                # LOGGER.info("n_time =", combined_jones.shape[2])

                if distance_from_identity < 1e-3:
                    LOGGER.warning(
                        "Combined Jones matrix is very close to identity! "
                        "Effects may be too small."
                    )

                # LOGGER.info(f"combined_jones: {combined_jones}")
                # Apply the combined atmospheric effect
                int_values = atmospheric_effects.apply_jones_to_visibilities(
                    int_values,  # [n_baselines, n_freq, 2, 2]
                    combined_jones,  # [n_ant, n_freq, n_time, 2, 2]
                    st1,
                    st2,
                    time_idx,
                )
                # int_values remains [n_baselines, n_freq, 2, 2]

            except Exception as e:
                LOGGER.error(f"Error applying atmospheric effects: {e}")
                LOGGER.info("Continuing without atmospheric effects for this batch")

        if noise_level != 0:
            noise, weights = generate_noise(
                int_values.shape,
                obs,
                noise_level,
                mode=noise_mode,
                el1_deg=bas_p.el1_valid,
                el2_deg=bas_p.el2_valid,
                telescope=telescope,
                band=band,
            )
            int_values += noise
        else:
            weights = torch.ones(int_values.shape[0], int_values.shape[1])
        vis_num = torch.arange(int_values.shape[0]) + 1 + vis_num.max()

        # LOGGER.info(f"Differenz orig - int_values: {(orig - int_values).abs().max()}")
        # Extract visibility components - back to 2D [n_baselines, n_freq]
        vis = Visibilities(
            V_11=int_values[..., 0, 0].cpu(),
            V_22=int_values[..., 1, 1].cpu(),
            V_12=int_values[..., 0, 1].cpu(),
            V_21=int_values[..., 1, 0].cpu(),
            weights=weights.cpu(),
            num=vis_num,
            base_num=bas_p.baseline_nums[~int_values_nans].cpu(),
            u=bas_p.u_valid[~int_values_nans].cpu(),
            v=bas_p.v_valid[~int_values_nans].cpu(),
            w=bas_p.w_valid[~int_values_nans].cpu(),
            date=bas_p.date[~int_values_nans].cpu(),
            st_id_pairs=bas_p.st_id_pairs[~int_values_nans].cpu(),
            linear_dop=torch.tensor([]),
            circular_dop=torch.tensor([]),
        )

        visibilities.add(vis)
        del int_values

    return visibilities


class AtmosphericEffects:
    """Simulate atmospheric effects in the Jones formalism.

    This class implements various atmospheric propagation effects including:
    - Ionospheric delay
    (- Faraday rotation)
    - Tropospheric delay

    Parameters
    ----------
    obs : Observation
        : class:`~pyvisgen.simulation.Observation` object containing
        observation parameters.
    n_time : int
        Number of time samples.
    time_axis : torch. Tensor
        Time array with shape [n_time] in seconds.

    Attributes
    ----------
    c : float
        Speed of light in m/s.
    earth_rotation_rate : float
        Earth rotation rate in rad/s.
    n_ant : int
        Number of antennas from observation.
    n_freq : int
        Number of frequency channels from observation.
    jones_effects : list
        List of Jones matrices for each effect.
    effect_names : list
        Names of atmospheric effects.
    """

    def __init__(
        self,
        obs,
        n_time: int,
        time_axis: torch.Tensor,
    ) -> None:
        self.obs = obs
        self.n_time = n_time
        self.time_axis = time_axis.to(obs.device)
        self.device = obs.device

        # Extract from observation
        self.n_ant = len(obs.array.st_num)
        self.n_freq = len(obs.waves_low)

        # Get frequencies from observation (center frequencies)
        self.frequencies = (
            torch.tensor(obs.waves_low, device=obs.device)
            + torch.tensor(obs.waves_high, device=obs.device)
        ) / 2.0

        # Physical constants from scipy and astropy
        self.c = const.c
        self.earth_rotation_rate = (
            astro_const.GM_earth.value / astro_const.R_earth.value**3
        ) ** 0.5
        self.earth_rotation_rate = (
            (2.0 * np.pi * un.rad / (1 * un.sday)).to(un.rad / un.s).value
        )

        # Extract antenna positions from observation
        self._extract_antenna_positions()

        # Storage for Jones effects
        self.jones_effects = []
        self.effect_names = []

    def _extract_antenna_positions(self) -> None:
        """Extract antenna positions from Observation object."""
        from pyvisgen.layouts import get_array_layout

        layout = get_array_layout(self.obs.layout)

        # Extract geocentric positions [n_ant, 3]
        self.ant_positions = torch.stack(
            [
                torch.tensor(layout.x, dtype=torch.float64, device=self.device),
                torch.tensor(layout.y, dtype=torch.float64, device=self.device),
                torch.tensor(layout.z, dtype=torch.float64, device=self.device),
            ],
            dim=1,
        )

    def _add_jones_effect(
        self,
        jones_matrix: torch.Tensor,
        effect_name: str,
    ) -> None:
        """Add a Jones matrix effect to the collection.

        Parameters
        ----------
        jones_matrix : torch.Tensor
            Jones matrix with shape [n_ant, n_freq, n_time, 2, 2].
        effect_name :  str
            Name of the atmospheric effect.
        """
        self.jones_effects.append(jones_matrix)
        self.effect_names.append(effect_name)
        LOGGER.info(f"Added Jones effect: {effect_name}")
        return self

    def get_all_jones_effects(self) -> torch.Tensor:
        """Get all Jones effects stacked together.

        Returns
        -------
        torch.Tensor
            Stacked Jones matrices with shape [n_effects, n_ant, n_freq, n_time, 2, 2].
        """
        if len(self.jones_effects) == 0:
            # Return identity matrices if no effects
            identity = torch.eye(2, dtype=torch.complex128, device=self.device).reshape(
                1, 1, 1, 1, 2, 2
            )
            return identity.expand(1, self.n_ant, self.n_freq, self.n_time, 2, 2)

        return torch.stack(self.jones_effects, dim=0)

    def _saastamoinen_zhd(
        self,
        pressure_hpa: float,
        latitude_deg: float,
        height_m: float,
        device,
        dtype=torch.float64,
    ) -> torch.Tensor:
        """Zenith hydrostatic delay in meters."""
        lat = torch.deg2rad(torch.tensor(latitude_deg, device=device, dtype=dtype))
        h_km = torch.tensor(height_m / 1000.0, device=device, dtype=dtype)

        return (
            0.0022768
            * torch.tensor(pressure_hpa, device=device, dtype=dtype)
            / (1.0 - 0.00266 * torch.cos(2.0 * lat) - 0.00028 * h_km)
        )

    def _approx_zwd_from_surface_met(
        self,
        temperature_K: float,
        relative_humidity: float,
        device,
        dtype=torch.float64,
    ) -> torch.Tensor:
        """Approximate zenith wet delay in meters from surface meteorology."""
        T = torch.tensor(temperature_K, device=device, dtype=dtype)
        RH = torch.tensor(relative_humidity / 100.0, device=device, dtype=dtype)

        # Tetens saturation vapor pressure, hPa
        T_C = T - 273.15
        e_s = 6.112 * torch.exp((17.62 * T_C) / (243.12 + T_C))
        e = RH * e_s

        # Common approximate ZWD formula, meters
        zwd = 0.002277 * (1255.0 / T + 0.05) * e
        return zwd

    def _mapping_continued_fraction(self, el_rad, a, b, c):
        """Continued-fraction mapping function."""
        s = torch.sin(el_rad).clamp_min(1e-3)
        numerator = 1.0 + a / (1.0 + b / (1.0 + c))
        denominator = s + a / (s + b / (s + c))
        return numerator / denominator

    def _simple_niell_like_mapping(
        self, elevation_angle, latitude_deg, device, dtype=torch.float64
    ):
        """Approximate Niell-style hydrostatic/wet mapping functions."""
        el = torch.deg2rad(torch.tensor(elevation_angle, device=device, dtype=dtype))

        # Niell-like coefficient tables versus |latitude|.
        # Values are for simple interpolation and practical simulation use.
        lats = [15.0, 30.0, 45.0, 60.0, 75.0]

        ah = [1.2769934e-3, 1.2683230e-3, 1.2465397e-3, 1.2196049e-3, 1.2045996e-3]
        bh = [2.9153695e-3, 2.9152299e-3, 2.9288445e-3, 2.9022565e-3, 2.9024912e-3]
        ch = [62.610505e-3, 62.837393e-3, 63.721774e-3, 63.824265e-3, 64.258455e-3]

        aw = [5.8021897e-4, 5.6794847e-4, 5.8118019e-4, 5.9727542e-4, 6.1641693e-4]
        bw = [1.4275268e-3, 1.5138625e-3, 1.4572752e-3, 1.5007428e-3, 1.7599082e-3]
        cw = [4.3472961e-2, 4.6729510e-2, 4.3908931e-2, 4.4626982e-2, 5.4736038e-2]

        def interp_lat(table):
            x = max(15.0, min(75.0, abs(latitude_deg)))
            for i in range(len(lats) - 1):
                if lats[i] <= x <= lats[i + 1]:
                    w = (x - lats[i]) / (lats[i + 1] - lats[i])
                    return table[i] * (1.0 - w) + table[i + 1] * w
            return table[-1]

        ah_t = torch.tensor(interp_lat(ah), device=device, dtype=dtype)
        bh_t = torch.tensor(interp_lat(bh), device=device, dtype=dtype)
        ch_t = torch.tensor(interp_lat(ch), device=device, dtype=dtype)

        aw_t = torch.tensor(interp_lat(aw), device=device, dtype=dtype)
        bw_t = torch.tensor(interp_lat(bw), device=device, dtype=dtype)
        cw_t = torch.tensor(interp_lat(cw), device=device, dtype=dtype)

        m_h = self._mapping_continued_fraction(el, ah_t, bh_t, ch_t)
        m_w = self._mapping_continued_fraction(el, aw_t, bw_t, cw_t)

        return m_h, m_w

    def _calc_elevation_deg(self, obs, time, dtype=torch.float64) -> torch.Tensor:
        """Compute source elevation for all antennas and time steps.

        Returns
        -------
        torch.Tensor
            Elevation angles in degrees with shape [n_ant, n_time].
        """

        src_crd = SkyCoord(
            ra=obs.ra.detach().numpy(),
            dec=obs.dec.detach().numpy(),
            unit=(un.deg, un.deg),
        )

        ant_locs = obs.array_earth_loc
        n_time = len(time)

        x = ant_locs.x.to_value(un.m)
        y = ant_locs.y.to_value(un.m)
        z = ant_locs.z.to_value(un.m)

        x_grid = np.repeat(x[None, :], n_time, axis=0)
        y_grid = np.repeat(y[None, :], n_time, axis=0)
        z_grid = np.repeat(z[None, :], n_time, axis=0)

        loc_grid = EarthLocation.from_geocentric(
            x_grid * un.m,
            y_grid * un.m,
            z_grid * un.m,
        )

        altaz = src_crd.transform_to(
            AltAz(
                obstime=time[..., None],
                location=loc_grid,
            )
        )
        LOGGER.info(f"altaz shape: {altaz.shape}, altaz.alt.shape: {altaz.alt.shape}")
        # Astropy gives [n_time, n_ant], but we need [n_ant, n_time]
        elevation_deg = altaz.alt.degree.T

        return torch.as_tensor(
            elevation_deg,
            device=self.device,
            dtype=dtype,
        )

    def _calc_azimuth_deg(
        self,
        obs,
        time,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        """Compute geometric source azimuth for all antennas and times.

        Parameters
        ----------
        obs
            Must provide scalar ICRS source coordinates ``ra`` and ``dec``
            in degrees and ``array_earth_loc`` as absolute EarthLocation
            coordinates.
        time : astropy.time.Time
            One-dimensional observation times.

        Returns
        -------
        torch.Tensor
            Azimuth in degrees, shape [n_ant, n_time].
            Convention: north=0 deg, east=90 deg.
        """

        if not isinstance(time, Time):
            raise TypeError("time must be an astropy.time.Time object")

        # Normalize scalar time to a one-element time axis.
        times = time[np.newaxis] if time.isscalar else time

        if times.ndim != 1:
            raise ValueError("time must be scalar or one-dimensional")

        if not isinstance(obs.array_earth_loc, EarthLocation):
            raise TypeError("obs.array_earth_loc must be an EarthLocation")

        ant_locs = obs.array_earth_loc
        if ant_locs.isscalar:
            ant_locs = ant_locs[np.newaxis]

        # Astropy/NumPy requires CPU data. This intentionally breaks autograd.
        ra = obs.ra.detach().cpu().numpy()
        dec = obs.dec.detach().cpu().numpy()

        if np.size(ra) != 1 or np.size(dec) != 1:
            raise ValueError("This function expects exactly one source coordinate")

        src_crd = SkyCoord(
            ra=np.asarray(ra).item() * un.deg,
            dec=np.asarray(dec).item() * un.deg,
            frame="icrs",
        )

        # Shapes:
        # obstime: [n_time, 1]
        # location: [1, n_ant]
        # resulting AltAz: [n_time, n_ant]
        frame = AltAz(
            obstime=times[:, np.newaxis],
            location=ant_locs[np.newaxis, :],
            pressure=0 * un.hPa,
        )

        az = src_crd.transform_to(frame).az.to_value(un.deg).T

        # Contiguous representation is predictable when converting to Torch.
        az = np.ascontiguousarray(az)

        return torch.as_tensor(
            az,
            device=self.device,
            dtype=dtype,
        )

    def simulate_ionospheric_delay(
        self,
        tec_values: torch.Tensor,
    ) -> torch.Tensor:
        """Simulate ionospheric time delay (dispersive).

        Parameters
        ----------
        tec_values : torch. Tensor
            Total Electron Content with shape [n_ant, n_time] in TECU
            (1 TECU = 10^16 electrons/m^2).

        Returns
        -------
        torch.Tensor
            Jones matrix with shape [n_ant, n_freq, n_time, 2, 2].
        """
        # Calculate ionospheric constant from fundamental constants
        K_iono = const.e**2 / (8 * np.pi**2 * const.epsilon_0 * const.m_e)

        # Expand dimensions for broadcasting
        tec_expanded = tec_values.unsqueeze(1)  # [n_ant, 1, n_time]
        tec_expanded *= 1e16  # Convert from TECU to electrons/m^3
        freq_expanded = self.frequencies.unsqueeze(0).unsqueeze(2)  # [1, n_freq, 1]

        # Ionospheric delay
        delay_meters = K_iono * tec_expanded / freq_expanded**2
        delay_seconds = delay_meters / self.c

        # Convert to Jones matrix
        jones_matrix = self._delay_to_jones_phase(delay_seconds)

        # Add to collection
        self._add_jones_effect(jones_matrix, "ionospheric_delay")
        LOGGER.info("Ionospheric delay simulated.")

        return self.get_all_jones_effects()

    def simulate_tropospheric_delay(
        self,
        obs,
        pressure_hpa: float = 1013.25,
        temperature_K: float = 288.15,
        relative_humidity: float = 50.0,
    ) -> torch.Tensor:
        """Simulate the tropospheric phase delay.

        Uses:
        - Saastamoinen zenith hydrostatic delay
        - approximate zenith wet delay from surface meteorology
        - time- and antenna-dependent elevation angles
        - approximate Niell-like hydrostatic and wet mapping functions

        Returns
        -------
        torch.Tensor
            Jones matrix with shape [n_ant, n_freq, n_time, 2, 2].
        """
        dtype = torch.float64

        ant_locs = obs.array_earth_loc

        latitude_deg = ant_locs.lat.to_value(un.deg).ravel()

        times_sec, times_jd, times = self.timesteps(obs)

        height_m = ant_locs.height.to_value(un.m).ravel()

        n_ant = self.ant_positions.shape[0]
        n_time = self.time_axis.shape[0]

        zhd_m = self._saastamoinen_zhd(
            pressure_hpa=pressure_hpa,
            latitude_deg=latitude_deg,
            height_m=height_m,
            device=self.device,
            dtype=dtype,
        )

        zwd_m = self._approx_zwd_from_surface_met(
            temperature_K=temperature_K,
            relative_humidity=relative_humidity,
            device=self.device,
            dtype=dtype,
        )

        elevation_angle = self._calc_elevation_deg(obs=obs, time=times)

        if elevation_angle.shape != (n_ant, n_time):
            raise ValueError(
                f"Expected elevation shape ({n_ant}, {n_time}), "
                f"got {tuple(elevation_angle.shape)}."
            )

        latitude_ref_deg = torch.as_tensor(latitude_deg).flatten()[0].item()

        m_h, m_w = self._simple_niell_like_mapping(
            elevation_angle=elevation_angle,
            latitude_deg=latitude_ref_deg,
            device=self.device,
            dtype=dtype,
        )

        zhd_m = torch.as_tensor(
            zhd_m,
            device=self.device,
            dtype=dtype,
        )

        if zhd_m.ndim == 1:
            zhd_m = zhd_m[:, None]

        zwd_m = torch.as_tensor(
            zwd_m,
            device=self.device,
            dtype=dtype,
        )

        if zwd_m.ndim == 0:
            zwd_m = zwd_m[None, None]

        path_delay_m = m_h * zhd_m + m_w * zwd_m

        total_delay_s = path_delay_m / self.c

        jones_matrix = self._delay_to_jones_phase(total_delay_s)

        self._add_jones_effect(jones_matrix, "tropospheric_delay")

        LOGGER.info(
            "Tropospheric delay simulated: %.3f m dry, %.3f m wet, %.3f m total slant.",
            zhd_m.mean().item(),
            zwd_m.mean().item(),
            path_delay_m.mean().item(),
        )

        return self.get_all_jones_effects()

    def _delay_to_jones_phase(
        self,
        delay: torch.Tensor,
        polarization_dependent: bool = False,
        polarization_offset: float = 0.0,
    ) -> torch.Tensor:
        """Convert time delay to Jones matrix phases.


        Parameters
        ----------
        delay : torch.Tensor
            Time delay in seconds with shape [n_ant, n_time]
            or [n_ant, n_freq, n_time].
        polarization_dependent : bool, optional
            Whether V_11/V_22 polarizations have different phases.  Default: False
        polarization_offset : float, optional
            Phase difference between V_11 and V_22 in radians. Default: 0.0

        Returns
        -------
        torch.Tensor
            Jones matrices as exp(i phi) with shape [n_ant, n_freq, n_time, 2, 2].
        """
        # Expand frequency dimension if needed
        if delay.ndim == 2:  # [n_ant, n_time]
            delay = delay.unsqueeze(1)  # [n_ant, 1, n_time]

        # Compute phase
        freq_expanded = self.frequencies.unsqueeze(0).unsqueeze(2)  # [1, n_freq, 1]
        phase = 2.0 * np.pi * freq_expanded * delay  # [n_ant, n_freq, n_time]

        # Jones components
        if polarization_dependent:
            phase_11 = phase
            phase_22 = phase + polarization_offset
        else:
            phase_11 = phase
            phase_22 = phase

        # Create Jones matrix [n_ant, n_freq, n_time, 2, 2]
        jones_matrix = torch.zeros(
            *phase.shape, 2, 2, dtype=torch.complex128, device=self.device
        )

        # Diagonal elements (phase delays)
        jones_matrix[..., 0, 0] = torch.exp(1j * phase_11.to(torch.complex128))
        jones_matrix[..., 1, 1] = torch.exp(1j * phase_22.to(torch.complex128))

        # Off-diagonal elements are zero for pure phase delay
        jones_matrix[..., 0, 1] = 0.0
        jones_matrix[..., 1, 0] = 0.0

        return jones_matrix

    def simulate_faraday_rotation(
        self,
        tec_values: torch.Tensor,
        magnetic_field_parallel: float = 0.3e-4,
    ) -> torch.Tensor:
        """Simulate Faraday rotation (ionospheric effect).

        Parameters
        ----------
        tec_values : torch. Tensor
            TEC values with shape [n_ant, n_time] in TECU.
        magnetic_field_parallel : float, optional
            Parallel component of Earth's magnetic field in Tesla.
            Default: 0.3e-4 (typical mid-latitude value).

        Returns
        -------
        torch. Tensor
            Jones matrix with shape [n_ant, n_freq, n_time, 2, 2].
        """
        # Calculate Faraday constant from fundamental constants
        K_faraday = (
            const.e**3
            / (8 * np.pi**2 * const.epsilon_0 * const.m_e**2 * const.c)
            * 1e16  # Convert from Tecu to electrons/m^2
        )

        tec_expanded = tec_values.unsqueeze(1)  # [n_ant, 1, n_time]
        freq_expanded = self.frequencies.unsqueeze(0).unsqueeze(2)  # [1, n_freq, 1]

        rotation_angle = (
            K_faraday * magnetic_field_parallel * tec_expanded / freq_expanded**2
        )  # [n_ant, n_freq, n_time]

        # Jones components
        cos_omega = torch.cos(rotation_angle)
        sin_omega = torch.sin(rotation_angle)

        # Create Jones matrix [n_ant, n_freq, n_time, 2, 2]
        jones_matrix = torch.zeros(
            *rotation_angle.shape, 2, 2, dtype=torch.complex128, device=self.device
        )

        jones_matrix[..., 0, 0] = cos_omega.to(torch.complex128)
        jones_matrix[..., 1, 1] = cos_omega.to(torch.complex128)
        jones_matrix[..., 0, 1] = -sin_omega.to(torch.complex128)
        jones_matrix[..., 1, 0] = sin_omega.to(torch.complex128)

        # Add to collection
        self._add_jones_effect(jones_matrix, "faraday_rotation")
        LOGGER.info("Faraday rotation simulated.")
        return self.get_all_jones_effects()

    def apply_jones_to_visibilities(
        self,
        int_values: torch.Tensor,
        jones: torch.Tensor,
        st1: torch.Tensor,
        st2: torch.Tensor,
        time_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Apply Jones matrices to baseline visibilities using the RIME.

        For a baseline (i,j), the corrupted visibility is:
        V'_ij = J_i * V_ij * J_j^H

        Parameters
        ----------
        int_values : torch.Tensor
            Input visibilities with shape [n_baselines, n_freq, 2, 2].
        jones : torch.Tensor
            Jones matrices with shape [n_ant, n_freq, n_time, 2, 2].
        st1 : torch.Tensor
            Station 1 indices for each baseline with shape [n_baselines].
        st2 : torch.Tensor
            Station 2 indices for each baseline with shape [n_baselines].
        time_idx : torch.Tensor
            [n_baselines], integer indices in [0, n_time-1]
        Returns
        -------
        torch.Tensor
            Corrupted visibilities with shape [n_baselines, n_freq, 2, 2].

        Notes
        -----
        This implements the full RIME equation:
        V'_pq = J_p V_pq J_q^H

        where p and q are antenna indices (st1 and st2).
        """
        # # Assume single time (t=0) for now
        # jones_single_time = jones[:, :, 0, :, :]  # [n_ant, n_freq, 2, 2]

        # # Extract Jones matrices for each baseline
        # jones_i = jones_single_time[st1.long()]  # [n_baselines, n_freq, 2, 2]
        # jones_j = jones_single_time[st2.long()]  # [n_baselines, n_freq, 2, 2]

        # # Ensure consistent dtype - convert int_values to match jones dtype
        # if int_values.dtype != jones_i.dtype:
        #     int_values = int_values.to(jones_i.dtype)

        st1 = st1.long()
        st2 = st2.long()
        time_idx = time_idx.long()

        if int_values.dtype != jones.dtype:
            int_values = int_values.to(jones.dtype)

        n_baselines, n_freq = int_values.shape[:2]

        # Reorder to [n_ant, n_time, n_freq, 2, 2]
        jones_tf = jones.permute(0, 2, 1, 3, 4)

        # Select Jones per baseline and per time
        jones_i = jones_tf[st1, time_idx]  # [n_baselines, n_freq, 2, 2]
        jones_j = jones_tf[st2, time_idx]  # [n_baselines, n_freq, 2, 2]

        # Hermitian conjugate for antenna j
        jones_j_herm = torch.conj(jones_j.transpose(-2, -1))

        # Apply RIME:  V' = J_i @ V @ J_j^H
        corrupted_vis = torch.einsum(
            "bfij,bfjk,bfkl->bfil", jones_i, int_values, jones_j_herm
        )
        return corrupted_vis

    def tec_field_from_iri(obs, return_times=bool):
        """Generate TEC field from IRI (International Reference Ionosphere) for given observation.
        Returns
        -------
        tec_field : torch.Tensor [n_ant, n_time] in TECU
        (optional) times : astropy.time.Time [n_time]
        """

        from iricore import iri

        # antenna positions
        ant_locs = obs.array_earth_loc
        alat = ant_locs.lat.to_value(un.deg).ravel()
        alon = ant_locs.lon.to_value(un.deg).ravel()
        n_ant = len(alat)

        # mid integration time for each scans tec values
        times_list = []

        for scan in obs.scans:
            start = scan.start
            stop = scan.stop
            int_time = scan.integration_time

            dur_s = (stop - start).to_value(un.s)
            int_s = int_time.to_value(un.s)
            n_int = int(np.floor(dur_s / int_s))
            if n_int <= 0:
                continue

            offsets = (np.arange(n_int) + 0.5) * int_s
            t_mid = start + offsets * un.s
            times_list.append(t_mid.utc.jd)

        if len(times_list) == 0:
            LOGGER.error("tec_field_from_iri: No times produced from obs.scans.")

        times = Time(np.concatenate(times_list), format="jd", scale="utc")
        n_time = len(times)

        alt0, alt1, dalt = 90.0, 2000.0, 2.0  # km
        alt_km = np.arange(alt0, alt1 + dalt, dalt, dtype=float)
        alt_m = alt_km * 1000.0

        tec_np = np.empty((n_ant, n_time), dtype=np.float64)

        # Loop over time and antenna
        for it, t in enumerate(times.utc.datetime):
            # iricore expects UTC naive
            dt = t.replace(tzinfo=None)

            for ia in range(n_ant):
                out = iri(
                    dt, [alt0, alt1, dalt], float(alat[ia]), float(alon[ia]), version=20
                )

                ne = np.asarray(out.edens, dtype=float)
                if np.any(ne < 0) or np.all(ne == 0):
                    tec_np[ia, it] = np.nan
                    continue
                vtec_e_m3 = np.trapezoid(ne, alt_m)
                tec_np[ia, it] = vtec_e_m3 / 1e16  # in TECU

        tec_field = torch.tensor(tec_np, dtype=torch.float64, device=obs.device)

        if return_times:
            return tec_field, times
        return tec_field

    def slant_tec_from_iri(self, obs):
        """Generate TEC field from IRI (International Reference Ionosphere) for given observation.
        Returns
        -------
        tec_field : torch.Tensor [n_ant, n_time] in TECU
        """

        from iricore import stec

        # antenna positions
        ant_locs = obs.array_earth_loc
        alat = ant_locs.lat.to_value(un.deg).ravel()
        alon = ant_locs.lon.to_value(un.deg).ravel()
        n_ant = len(alat)

        times_sec, times_jd, times = self.timesteps(obs)

        n_time = len(times)
        el = self._calc_elevation_deg(obs, times.utc.datetime)
        az = self._calc_azimuth_deg(obs, times)

        # iricore uses NumPy and therefore requires CPU data
        if isinstance(el, torch.Tensor):
            el = el.detach().cpu().numpy()
        else:
            el = np.asarray(el)

        if isinstance(az, torch.Tensor):
            az = az.detach().cpu().numpy()
        else:
            az = np.asarray(az)

        tec_np = np.empty((n_ant, n_time), dtype=np.float64)

        # Loop over time and antenna
        for it, t in enumerate(times.utc.datetime):
            # iricore expects UTC naive
            dt = t.replace(tzinfo=None)

            for ia in range(n_ant):
                out = stec(
                    el=el[ia, it],
                    az=az[ia, it],
                    dt=dt,
                    lat=alat[ia],
                    lon=alon[ia],
                    version=20,
                    npoints=40,
                )
                tec_np[ia, it] = out  # in TECU

        tec_field = torch.tensor(tec_np, dtype=torch.float64, device=obs.device)

        return tec_field

    def timesteps(self, obs):
        """Return time axis in seconds since first integration."""
        times_list = []

        for scan in obs.scans:
            start = scan.start
            stop = scan.stop
            int_time = scan.integration_time

            dur_s = (stop - start).to_value(un.s)
            int_s = int_time.to_value(un.s)
            n_int = int(np.floor(dur_s / int_s))
            if n_int <= 0:
                continue

            offsets = (np.arange(n_int) + 0.5) * int_s
            t_mid = start + offsets * un.s
            times_list.append(t_mid.utc.jd)

        if len(times_list) == 0:
            LOGGER.error("timesteps: No integration times produced from obs.scans.")

        times = Time(np.concatenate(times_list), format="jd", scale="utc")

        time_axis_jd = torch.tensor(times.jd, dtype=torch.float64, device=obs.device)
        time_axis_sec = torch.tensor(
            (times.jd - times.jd[0]) * 86400.0,
            dtype=torch.float64,
            device=obs.device,
        )
        return time_axis_sec, time_axis_jd, times
