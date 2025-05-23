from dataclasses import dataclass, fields

import scipy.ndimage
import toma
import torch
from tqdm.auto import tqdm

import pyvisgen.simulation.scan as scan

torch.set_default_dtype(torch.float64)

__all__ = [
    "Visibilities",
    "vis_loop",
    "Polarization",
    "generate_noise",
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
        mask = (self.SI >= self.sensitivity_cut)[..., 0]
        B = B[mask]

        return B, mask, self.lin_dop, self.circ_dop

    def rand_polarization_field(
        self,
        shape: list[int, int] | int,
        order: list[int, int] | int = 1,
        random_state: int = None,
        scale: list = [0, 1],
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
) -> Visibilities:
    r"""Computes the visibilities of an observation.

    Parameters
    ----------
    obs : Observation class object
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
    mode : str, optional
        Select one of `'full'`, `'grid'`, or `'dense'` to get
        all valid baselines, a grid of unique baselines, or
        dense baselines. Default: 'full'
    batch_size : int, optional
        Batch size for iteration over baselines. Default: 100
    polarization : str, optional
        Choose between `'linear'` or `'circular'` or `None` to
        simulate different types of polarizations or disable
        the simulation of polarization. Default: 'linear'
    random_state : int, optional
        Random state used when drawing `amp_ratio` and during the generation
        of the random polarization field. Default: 42
    show_progress : bool, optional
        If `True`, show a progress bar during the iteration over the
        batches of baselines. Default: False
    normalize : bool, optional
        If ``True``, normalize stokes matrix ``B`` by a factor 0.5.
        Default: ``True``

    Returns
    -------
    visibilities : Visibilities
        Dataclass object containing visibilities and baselines.
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
        obs.calc_dense_baselines()
        bas = obs.dense_baselines_gpu
    else:
        raise ValueError("Unsupported mode!")

    if batch_size == "auto":
        batch_size = bas[:].shape[1]

    visibilities = toma.explicit.batch(
        _batch_loop,
        batch_size,
        visibilities,
        vis_num,
        obs,
        B,
        bas,
        lm,
        rd,
        noisy,
        show_progress,
        mode,
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
):
    """Main simulation loop of pyvisgen. Computes visibilities
    batchwise.

    Parameters
    ----------
    batch_size : int
        Batch size for loop over Baselines dataclass object.
    visibilities : Visibilities
        Visibilities dataclass object.
    vis_num : int
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
    noisy : float or bool
        Simulate noise as SEFD with given value. If set to False,
        no noise is simulated.
    show_progress :
        If True, show a progress bar tracking the loop.

    Returns
    -------
    visibilities : Visibilities
        Visibilities dataclass object.
    """
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
                )[None]
                for wave_low, wave_high in zip(obs.waves_low, obs.waves_high)
            ]
        )
        if int_values.numel() == 0:
            continue

        int_values = torch.swapaxes(int_values, 0, 1)

        if noisy != 0:
            noise = generate_noise(int_values.shape, obs, noisy)
            int_values += noise

        vis_num = torch.arange(int_values.shape[0]) + 1 + vis_num.max()

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
