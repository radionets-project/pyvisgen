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
from astropy.coordinates import EarthLocation

from typing import Optional


import torch.nn.functional as F
import pyvisgen.simulation.scan as scan
from pyvisgen.utils.batch_size import adaptive_batch_size
from pyvisgen.utils.logging import setup_logger


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
    "generate_tec_field",
    "tec_field_from_iri",
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
    num : :func:`~torch.tensor`
    base_num : :func:`~torch.tensor`
    u : :func:`~torch.tensor`
    v : :func:`~torch.tensor`
    w  : :func:`~torch.tensor`
    date : :func:`~torch.tensor`
    linear_dop : :func:`~torch.tensor`
    circular_dop : :func:`~torch.tensor`
    """

    V_11: torch.tensor
    V_22: torch.tensor
    V_12: torch.tensor
    V_21: torch.tensor
    num: torch.tensor
    base_num: torch.tensor
    u: torch.tensor
    v: torch.tensor
    w: torch.tensor
    date: torch.tensor
    linear_dop: torch.tensor
    circular_dop: torch.tensor

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
        SI: torch.tensor,
        sensitivity_cut: float,
        amp_ratio: float,
        delta: float,
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

            if amp_ratio and (amp_ratio >= 0):
                ax2 = amp_ratio
            else:
                ax2 = torch.rand(1).to(self.device)

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
        shape: list[int, int] | int,
        order: list[int, int] | int = 1,
        random_state: int = None,
        scale: list = None,
        threshold: float = None,
    ) -> torch.tensor:
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
            torch.random.manual_seed(random_state)

        if isinstance(shape, int):
            shape = [shape]

        if not isinstance(shape, list):
            shape = list(shape)

        if len(shape) < 2:
            shape *= 2
        elif len(shape) > 2:
            raise ValueError("Only 2d shapes are allowed!")

        if isinstance(order, int):
            order = [order]

        if not isinstance(order, list):
            order = list(order)

        if len(order) < 2:
            order *= 2
        elif len(order) > 2:
            raise ValueError("Only 2d shapes are allowed!")

        sigma = torch.mean(torch.tensor(shape).double()) / (40 * torch.tensor(order))

        im = torch.rand(shape)
        im = scipy.ndimage.gaussian_filter(im, sigma=sigma.numpy())

        if scale is None:
            scale = [im.min(), im.max()]

        im_flatten = torch.from_numpy(im.flatten())
        im_argsort = torch.argsort(torch.argsort(im_flatten))
        im_linspace = torch.linspace(*scale, im_argsort.size()[0])
        uniform_flatten = im_linspace[im_argsort]

        im = torch.reshape(uniform_flatten, im.shape)

        if threshold:
            im = im < threshold

        return im


def vis_loop(
    obs,
    SI: torch.tensor,
    num_threads: int = 10,
    noisy: bool = True,
    mode: str = "full",
    batch_size: int = "auto",
    show_progress: bool = False,
    normalize: bool = True,
    ft: Literal["default", "finufft", "reversed"] = "default",
    atmospheric_effects: Optional["AtmosphericEffects"] = None,
    tec_values: Optional[torch.Tensor] = None,
    include_faraday: bool = True,
    magnetic_field_parallel: float = 0.3e-4,
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
    noisy : bool, optional
        If `True`, generate and add additional noise to
        the simulated measurements. Default: True
    mode :  str, optional
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
        Atmospheric effects simulator instance. If ``None``, no
        atmospheric effects are applied. Default: ``None``
    tec_values : torch.Tensor, optional
        TEC values with shape [n_ant, n_time] in TECU. Required if
        ``atmospheric_effects`` is not ``None``. Default: ``None``
    include_faraday : bool, optional
        Whether to include Faraday rotation in ionospheric effects.
        Only used if ``atmospheric_effects`` is not ``None``.
        Default: ``True``
    magnetic_field_parallel : float, optional
        Parallel component of Earth's magnetic field in Tesla.
        Only used if ``atmospheric_effects`` is not ``None``.
        Default: 0.3e-4

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

    if atmospheric_effects is not None and tec_values is None:
        raise ValueError(
            "tec_values must be provided when atmospheric_effects is not None"
        )
    if atmospheric_effects is None and tec_values is not None:
        LOGGER.warning(
            "tec_values provided but atmospheric_effects is None. "
            "No atmospheric effects will be applied."
        )

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
    if atmospheric_effects is not None and tec_values is not None:
        LOGGER.info("Computing atmospheric Jones matrices...")
        jones_atm = atmospheric_effects.simulate_ionospheric_delay(
            tec_values,
            include_faraday=include_faraday,
            magnetic_field_parallel=magnetic_field_parallel,
        )
        LOGGER.info(
            f"Atmospheric effects enabled:  {len(atmospheric_effects.effect_names)} effects"
        )

        # DEBUG: Check Jones matrices
        print(f"\n{'=' * 70}")
        print("DEBUG: Jones Matrix Statistics")
        print(f"{'=' * 70}")
        print(f"Jones shape: {jones_atm.shape}")
        for i, name in enumerate(atmospheric_effects.effect_names):
            jones_i = jones_atm[i]
            print(f"\nEffect {i}: {name}")
            print(f"  Mean magnitude: {torch.abs(jones_i).mean():.6f}")
            print(f"  Max magnitude: {torch.abs(jones_i).max():.6f}")
            print(f"  Min magnitude: {torch.abs(jones_i).min():.6f}")

            # Check if it's close to identity
            identity_test = jones_i[:, :, 0, :, :] - torch.eye(
                2, device=jones_i.device, dtype=jones_i.dtype
            )
            print(f"  Distance from identity: {torch.abs(identity_test).mean():.6e}")

            # Sample values
            print(f"  Sample J[0,0,0,:,:] = \n{jones_i[0, 0, 0, :, :]}")

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
            raise ValueError("Only available for GPU calculations!")

        obs.calc_dense_baselines()  # pragma: no cover
        bas = obs.dense_baselines_gpu  # pragma: nocover
    else:
        raise ValueError("Unsupported mode!")

    if batch_size == "auto":
        batch_size = bas[:].shape[1]

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
        noisy=noisy,
        show_progress=show_progress,
        mode=mode,
        ft=ft,
        atmospheric_effects=atmospheric_effects,
        jones_atm=jones_atm,
    )

    visibilities.linear_dop = lin_dop.cpu()
    visibilities.circular_dop = circ_dop.cpu()

    return visibilities


def _batch_loop(
    batch_size: int,
    visibilities,
    vis_num: int,
    obs,
    B: torch.tensor,
    bas,
    lm: torch.tensor,
    rd: torch.tensor,
    noisy: bool | float,
    show_progress: bool,
    mode: str,
    ft: Literal["default", "finufft", "reversed"] = "default",
    atmospheric_effects: Optional["AtmosphericEffects"] = None,
    jones_atm: Optional[torch.Tensor] = None,
):
    """Main simulation loop of pyvisgen. Computes visibilities batchwise."""

    batches = torch.arange(bas[:].shape[1]).split(batch_size)
    batches = tqdm(
        batches,
        position=0,
        disable=not show_progress,
        desc="Computing visibilities",
        postfix=f"Batch size: {batch_size}",
    )

    for p in batches:
        bas_p = bas[:][:, p]

        int_values = torch.cat(
            [
                scan.rime(
                    B,
                    bas_p,
                    lm,
                    rd,
                    obs.ra,
                    obs.dec,
                    torch.unique(obs.array.diam),
                    wave_low,
                    wave_high,
                    obs.polarization,
                    mode=mode,
                    corrupted=obs.corrupted,
                    ft=ft,
                )[None]
                for wave_low, wave_high in zip(obs.waves_low, obs.waves_high)
            ]
        )
        if int_values.numel() == 0:
            continue

        int_values = torch.swapaxes(int_values, 0, 1)
        orig = int_values.clone()
        # int_values shape: [n_baselines, n_freq, 2, 2]

        if atmospheric_effects is not None and jones_atm is not None:
            try:
                LOGGER.info("Applying atmospheric effects to visibilities...")
                # Extract antenna indices from baseline numbers
                base_num = bas_p[9]
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
                        LOGGER.info(f"Combining with effect {effect_idx}")

                        # DEBUG: Check before combination
                        before_magnitude = torch.abs(combined_jones).mean()

                        combined_jones = torch.matmul(
                            jones_atm[effect_idx], combined_jones
                        )

                        # DEBUG: Check after combination
                        after_magnitude = torch.abs(combined_jones).mean()
                        LOGGER.info(
                            f"  Magnitude before: {before_magnitude:.6e}, after: {after_magnitude:.6e}"
                        )

                # DEBUG: Check combined Jones matrix
                LOGGER.info(f"Combined Jones shape: {combined_jones.shape}")
                identity_test = combined_jones[:, :, 0, :, :] - torch.eye(
                    2, device=combined_jones.device, dtype=combined_jones.dtype
                )
                distance_from_identity = torch.abs(identity_test).mean()
                LOGGER.info(
                    f"Combined Jones distance from identity: {distance_from_identity:.6e}"
                )

                if distance_from_identity < 1e-10:
                    LOGGER.warning(
                        "Combined Jones matrix is very close to identity! Effects may be too small."
                    )

                # LOGGER.info(f"combined_jones: {combined_jones}")
                # Apply the combined atmospheric effect
                int_values = atmospheric_effects.apply_jones_to_visibilities(
                    int_values,  # [n_baselines, n_freq, 2, 2]
                    combined_jones,  # [n_ant, n_freq, n_time, 2, 2]
                    st1,
                    st2,
                )
                # int_values remains [n_baselines, n_freq, 2, 2]

            except Exception as e:
                LOGGER.error(f"Error applying atmospheric effects: {e}")
                LOGGER.info("Continuing without atmospheric effects for this batch")

        if noisy != 0:
            noise = generate_noise(int_values.shape, obs, noisy)
            int_values += noise
        vis_num = torch.arange(int_values.shape[0]) + 1 + vis_num.max()

        # LOGGER.info(f"Differenz orig - int_values: {(orig - int_values).abs().max()}")
        # Extract visibility components - back to 2D [n_baselines, n_freq]
        vis = Visibilities(
            int_values[..., 0, 0].cpu(),  # V_11
            int_values[..., 1, 1].cpu(),  # V_22
            int_values[..., 0, 1].cpu(),  # V_12
            int_values[..., 1, 0].cpu(),  # V_21
            vis_num,
            bas_p[9].cpu(),
            bas_p[2].cpu(),
            bas_p[5].cpu(),
            bas_p[8].cpu(),
            bas_p[10].cpu(),
            torch.tensor([]),
            torch.tensor([]),
        )

        visibilities.add(vis)
        del int_values

    return visibilities


def generate_noise(shape, obs, SEFD):
    # scaling factor for the noise
    factor = 1

    # system efficency factor, near 1
    eta = 0.93

    # taken from simulations
    chan_width = obs.bandwidths[0] * len(obs.bandwidths)

    # corr_int_time
    exposure = obs.int_time

    # taken from:
    # https://science.nrao.edu/facilities/vla/docs/manuals/oss/performance/sensitivity

    std = factor * 1 / eta * SEFD
    std /= torch.sqrt(2 * exposure * chan_width)
    noise = torch.normal(mean=0, std=std, size=shape, device=obs.device)
    noise = noise + 1.0j * torch.normal(mean=0, std=std, size=shape, device=obs.device)

    return noise


class AtmosphericEffects:
    """Simulate atmospheric effects in the Jones formalism.

    This class implements various atmospheric propagation effects including:
    - Ionospheric delay
    (- Faraday rotation)
    (- Tropospheric delay)

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
        print(f"Added Jones effect: {effect_name}")

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

    def simulate_tropospheric_delay(
        self,
        elevation_angle: float = 45.0,
        zenith_delay_wet: float = 0.1,
        zenith_delay_dry: float = 2.3,
        spatial_scale: float = 5000.0,
        temporal_scale: float = 600.0,
        random_state: Optional[int] = None,
    ) -> torch.Tensor:
        """Simulate tropospheric time delay.

        Parameters
        ----------
        elevation_angle : float, optional
            Source elevation in degrees. Default: 45.0
        zenith_delay_wet : float, optional
            Wet zenith component in meters. Default: 0.1
        zenith_delay_dry :  float, optional
            Dry zenith component in meters. Default: 2.3
        spatial_scale : float, optional
            Spatial correlation length in meters. Default: 5000.0
        temporal_scale :  float, optional
            Temporal correlation timescale in seconds. Default: 600.0
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        torch.Tensor
            Jones matrix with shape [n_ant, n_freq, n_time, 2, 2].
        """
        if random_state is not None:
            torch.manual_seed(random_state)

        # Mapping function (simple 1/sin(El) approximation)
        el_rad = torch.deg2rad(torch.tensor(elevation_angle, dtype=torch.float64))
        mapping_factor = 1.0 / torch.sin(el_rad)

        # Dry component (constant, spatially homogeneous)
        dry_delay = zenith_delay_dry * mapping_factor / self.c

        # Wet component (spatially and temporally variable)
        # Spatial covariance
        ant_pos_2d = self.ant_positions[:, :2]
        baseline_distances = torch.cdist(ant_pos_2d, ant_pos_2d)
        spatial_cov = torch.exp(-baseline_distances / spatial_scale)

        # Temporal covariance (exponential)
        time_diff = self.time_axis.unsqueeze(0) - self.time_axis.unsqueeze(1)
        temporal_cov = torch.exp(-torch.abs(time_diff) / temporal_scale)

        # Generate correlated noise (separable covariance)
        # Spatial component
        spatial_cov_reg = (
            spatial_cov
            + torch.eye(self.n_ant, device=self.device, dtype=torch.float64) * 1e-6
        )
        spatial_cholesky = torch.linalg.cholesky(spatial_cov_reg)
        spatial_noise = torch.randn(self.n_ant, dtype=torch.float64, device=self.device)
        spatial_component = torch.matmul(spatial_cholesky, spatial_noise)

        # Temporal component
        temporal_cov_reg = (
            temporal_cov
            + torch.eye(self.n_time, device=self.device, dtype=torch.float64) * 1e-6
        )
        temporal_cholesky = torch.linalg.cholesky(temporal_cov_reg)
        temporal_noise = torch.randn(
            self.n_time, dtype=torch.float64, device=self.device
        )
        temporal_component = torch.matmul(temporal_cholesky, temporal_noise)

        # Combine via outer product
        wet_delay_variation = torch.outer(spatial_component, temporal_component)
        wet_delay_variation *= zenith_delay_wet * mapping_factor / self.c

        # Total delay
        total_delay = dry_delay + wet_delay_variation

        # Convert to Jones matrix
        jones_matrix = self.delay_to_jones_phase(total_delay)

        # Add to collection
        self._add_jones_effect(jones_matrix, "tropospheric_delay")

        return jones_matrix

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
        K_iono = (
            const.e**2
            / (8 * np.pi**2 * const.epsilon_0 * const.m_e)
            * 1e16  # Convert from TECU to electrons/m^2
        )

        # Expand dimensions for broadcasting
        tec_expanded = tec_values.unsqueeze(1)  # [n_ant, 1, n_time]
        freq_expanded = self.frequencies.unsqueeze(0).unsqueeze(2)  # [1, n_freq, 1]

        # Ionospheric delay
        delay_meters = K_iono * tec_expanded / freq_expanded**2
        delay_seconds = delay_meters / self.c

        # Convert to Jones matrix
        jones_matrix = self.delay_to_jones_phase(delay_seconds)

        # Add to collection
        self._add_jones_effect(jones_matrix, "ionospheric_delay")
        LOGGER.info("Ionospheric delay simulated.")

        return self.get_all_jones_effects()

    def delay_to_jones_phase(
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

        # DEBUG OUTPUT
        LOGGER.info("DEBUG: Delay to Jones Phase Statistics")
        LOGGER.info(f"Frequencies: {self.frequencies / 1e6} MHz")
        LOGGER.info(f"Delay shape: {delay.shape}")
        LOGGER.info(f"Delay range: [{delay.min():.6e}, {delay.max():.6e}] seconds")
        LOGGER.info(f"Phase shape: {phase.shape}")
        LOGGER.info(f"Phase range: [{phase.min():.6e}, {phase.max():.6e}] radians")
        LOGGER.info(
            f"Phase range: [{torch.rad2deg(phase.min()):.6e}, {torch.rad2deg(phase.max()):.6e}] degrees"
        )
        LOGGER.info(f"Phase mean: {phase.mean():.6e} radians")

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
            * 1e-16  # Convert from electrons/m^2 to TECU
        )

        # Expand dimensions
        tec_expanded = tec_values.unsqueeze(1)  # [n_ant, 1, n_time]
        freq_expanded = self.frequencies.unsqueeze(0).unsqueeze(2)  # [1, n_freq, 1]

        # Rotation angle:
        rotation_angle = (
            K_faraday * magnetic_field_parallel * tec_expanded / freq_expanded**2
        )

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
        return jones_matrix

    def apply_jones_to_visibilities(
        self,
        int_values: torch.Tensor,
        jones: torch.Tensor,
        st1: torch.Tensor,
        st2: torch.Tensor,
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
        # Assume single time (t=0) for now
        jones_single_time = jones[:, :, 0, :, :]  # [n_ant, n_freq, 2, 2]

        # Extract Jones matrices for each baseline
        jones_i = jones_single_time[st1.long()]  # [n_baselines, n_freq, 2, 2]
        jones_j = jones_single_time[st2.long()]  # [n_baselines, n_freq, 2, 2]

        # Ensure consistent dtype - convert int_values to match jones dtype
        if int_values.dtype != jones_i.dtype:
            int_values = int_values.to(jones_i.dtype)

        # Hermitian conjugate for antenna j
        jones_j_herm = torch.conj(jones_j.transpose(-2, -1))

        # Apply RIME:  V' = J_i @ V @ J_j^H
        corrupted_vis = torch.einsum(
            "bfij,bfjk,bfkl->bfil", jones_i, int_values, jones_j_herm
        )
        print("applied jones to visibilities")
        return corrupted_vis


def generate_tec_field(
    n_ant: int,
    n_time: int,
    mean_tec: float = 10.0,
    std_tec: float = 2.0,
    spatial_scale: float = 50000.0,
    temporal_scale: float = 1800.0,
    device: Optional[torch.device] = None,
    random_state: Optional[int] = None,
) -> torch.Tensor:
    """Generate a realistic TEC field with spatial and temporal correlations.
    Parameters
    ----------
    n_ant : int
        Number of antennas.
    n_time : int
        Number of time samples.
    mean_tec : float, optional
        Mean TEC value in TECU. Default: 10.0
    std_tec : float, optional
        Standard deviation of TEC fluctuations in TECU. Default: 2.0
    spatial_scale : float, optional
        Spatial correlation length in meters. Default: 50000.0
    temporal_scale :  float, optional
        Temporal correlation timescale in seconds. Default: 1800.0
    device :  torch.device, optional
        PyTorch device.
    random_state : int, optional
        Random seed.
    Returns
    -------
    torch.Tensor
        TEC field with shape [n_ant, n_time] in TECU.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if random_state is not None:
        torch.manual_seed(random_state)
    # Start with mean TEC
    tec_field = torch.ones(n_ant, n_time, dtype=torch.float64, device=device) * mean_tec
    # Add correlated fluctuations
    fluctuations = torch.randn(n_ant, n_time, dtype=torch.float64, device=device)
    # Smooth fluctuations using 2D convolution
    kernel_size_spatial = max(3, int(n_ant / 10))
    kernel_size_temporal = max(3, int(n_time / 50))
    # Ensure odd kernel size
    if kernel_size_spatial % 2 == 0:
        kernel_size_spatial += 1
    if kernel_size_temporal % 2 == 0:
        kernel_size_temporal += 1
    kernel = torch.ones(
        1,
        1,
        kernel_size_spatial,
        kernel_size_temporal,
        dtype=torch.float64,
        device=device,
    ) / (kernel_size_spatial * kernel_size_temporal)
    fluctuations_smoothed = F.conv2d(
        fluctuations.unsqueeze(0).unsqueeze(0),
        kernel,
        padding=(kernel_size_spatial // 2, kernel_size_temporal // 2),
    ).squeeze()
    # Scale fluctuations
    fluctuations_smoothed = fluctuations_smoothed * std_tec
    tec_field += fluctuations_smoothed
    # Ensure TEC is always positive
    tec_field = torch.clamp(tec_field, min=0.1)

    return tec_field


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

        offsets = (np.arange(n_int) + 0.5) * int_s  # center of integration
        t_mid = start + offsets * un.s
        times_list.append(t_mid.utc.jd)

    if len(times_list) == 0:
        LOGGER.error(
            "tec_field_from_iri: No integration times produced from obs.scans."
        )

    times = Time(np.concatenate(times_list), format="jd", scale="utc")
    n_time = len(times)

    alt0, alt1, dalt = 90.0, 2000.0, 2.0  # km
    alt_km = np.arange(alt0, alt1 + dalt, dalt, dtype=float)
    alt_m = alt_km * 1000.0

    tec_np = np.empty((n_ant, n_time), dtype=np.float64)

    # Loop over time+antenna
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
            vtec_e_m2 = np.trapz(ne, alt_m)  # el/m^2
            tec_np[ia, it] = vtec_e_m2 / 1e16  # TECU

    tec_field = torch.tensor(tec_np, dtype=torch.float64, device=obs.device)

    if return_times:
        return tec_field, times
    return tec_field
