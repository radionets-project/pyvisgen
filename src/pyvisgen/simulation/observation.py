from dataclasses import dataclass, fields
from datetime import datetime

import astropy.units as un
import numpy as np
import numpy.typing
import torch
from astropy.constants import c
from astropy.coordinates import AltAz, EarthLocation, Longitude, SkyCoord
from astropy.time import Time
from tqdm.auto import tqdm

from pyvisgen.layouts import layouts
from pyvisgen.simulation.array import Array
from pyvisgen.utils.logging import setup_logger

torch.set_default_dtype(torch.float64)
LOGGER = setup_logger(namespace=__name__)

__all__ = ["Baselines", "Observation", "ValidBaselineSubset"]


DEFAULT_POL_KWARGS = {
    "delta": 0,
    "amp_ratio": 0.5,
    "random_state": 42,
}

DEFAULT_FIELD_KWARGS = {
    "order": [1, 1],
    "scale": [0, 1],
    "threshold": None,
    "random_state": 42,
}


@dataclass
class Baselines:
    """The Baselines dataclass comprises of data
    on station combinations, the u, v, and w coverage,
    validity of the measured data points (i.e. whether the
    source is visible for the antenna pairs, or not),
    observation time and parallactic angles for each
    baseline pair.

    Attributes
    ----------
    st1 : :func:`~torch.Tensor`
        Station IDs for antenna pairs.
    st2 : :func:`~torch.Tensor`
        Station IDs for antenna pairs.
    u : :func:`~torch.Tensor`
        u coordinate coverage.
    v : :func:`~torch.Tensor`
        v coordinate coverage.
    w : :func:`~torch.Tensor`
        w coordinate coverage.
    valid : :func:`~torch.Tensor`
        Mask of valid values, i.e. where the source
        is visible to the antenna pairs.
    time : :func:`~torch.Tensor`
        Tensor of observation time steps.
    q1 : :func:`~torch.Tensor`
        Tensor of parallactic angle values.
    q2 : :func:`~torch.Tensor`
        Tensor of parallactic angle values.
    el1 : :func:`~torch.Tensor`
        Tensor of elevations for ant 1
    el2 : :func:`~torch.Tensor`
        Tensor of elevations for ant 2
    """

    st1: torch.Tensor
    st2: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    w: torch.Tensor
    valid: torch.Tensor
    time: torch.Tensor
    q1: torch.Tensor
    q2: torch.Tensor
    el1: torch.Tensor
    el2: torch.Tensor

    def __getitem__(self, i):
        """Returns element at index ``i`` for all fields."""
        return Baselines(*[getattr(self, f.name)[i] for f in fields(self)])

    def add_baseline(self, baselines) -> None:
        """Adds a new baseline to the dataclass object.

        Parameters
        ----------
        baselines : :class:`~pyvisgen.simulation.Baselines`
            :class:`~pyvisgen.simulation.Baselines` dataclass object
            that is added to the fields of this dataclass.
        """
        [
            setattr(
                self,
                f.name,
                torch.cat([getattr(self, f.name), getattr(baselines, f.name)]),
            )
            for f in fields(self)
        ]

    def get_valid_subset(self, num_baselines: int, device: str):
        """Returns a valid subset of the baselines using
        the information stored in the ``valid`` field.

        Parameters
        ----------
        num_baselines : int
            Number of baselines used in the observation.
        device : str
            Name of the device to run the operation on,
            e.g. ``'cuda'`` or ``'cpu'``.

        Returns
        -------
        ValidBaselineSubset
            :class:`~pyvisgen.simulation.ValidBaselineSubset` dataclass
            object containing valid u, v, and w coverage, observation time
            steps, numbers of baselines, and parallactic angles.
        """
        bas_reshaped = Baselines(
            *[getattr(self, f.name).reshape(-1, num_baselines) for f in fields(self)]
        )

        mask = (bas_reshaped.valid[:-1].bool()) & (bas_reshaped.valid[1:].bool())

        # The baselines of all scans are stacked along the time axis. An integration
        # window spans two consecutive time steps, so the window between the last
        # time step of one scan and the first of the next one does not belong to
        # any measurement and has to be removed.
        scan_lengths = getattr(self, "scan_lengths", None)
        if scan_lengths is not None:
            first_step_of_next_scan = torch.cumsum(torch.tensor(scan_lengths), dim=0)[
                :-1
            ]
            mask[first_step_of_next_scan - 1] = False

        stations_1 = bas_reshaped.st1[:-1][mask].ravel()
        stations_2 = bas_reshaped.st2[:-1][mask].ravel()
        baseline_nums = (256 * (stations_1 + 1) + stations_2 + 1).to(device)

        st_id_pairs = torch.stack([stations_1, stations_2], dim=1).to(device)

        u_start = bas_reshaped.u[:-1][mask].to(device)
        v_start = bas_reshaped.v[:-1][mask].to(device)
        w_start = bas_reshaped.w[:-1][mask].to(device)

        u_stop = bas_reshaped.u[1:][mask].to(device)
        v_stop = bas_reshaped.v[1:][mask].to(device)
        w_stop = bas_reshaped.w[1:][mask].to(device)

        u_valid = (u_start + u_stop) / 2
        v_valid = (v_start + v_stop) / 2
        w_valid = (w_start + w_stop) / 2

        q1_start = bas_reshaped.q1[:-1][mask].to(device)
        q2_start = bas_reshaped.q2[:-1][mask].to(device)

        q1_stop = bas_reshaped.q1[1:][mask].to(device)
        q2_stop = bas_reshaped.q2[1:][mask].to(device)

        q1_valid = (q1_start + q1_stop) / 2
        q2_valid = (q2_start + q2_stop) / 2

        el1_start = bas_reshaped.el1[:-1][mask].to(device)
        el2_start = bas_reshaped.el2[:-1][mask].to(device)
        el1_stop = bas_reshaped.el1[1:][mask].to(device)
        el2_stop = bas_reshaped.el2[1:][mask].to(device)
        el1_valid = (el1_start + el1_stop) / 2
        el2_valid = (el2_start + el2_stop) / 2

        t = Time(bas_reshaped.time / (60 * 60 * 24), format="mjd").jd
        date = (torch.from_numpy(t[:-1][mask] + t[1:][mask]) / 2).to(device)

        return ValidBaselineSubset(
            u_start,
            u_stop,
            u_valid,
            v_start,
            v_stop,
            v_valid,
            w_start,
            w_stop,
            w_valid,
            baseline_nums,
            date,
            q1_start,
            q1_stop,
            q1_valid,
            q2_start,
            q2_stop,
            q2_valid,
            el1_start,
            el1_stop,
            el1_valid,
            el2_start,
            el2_stop,
            el2_valid,
            st_id_pairs,
        )


@dataclass()
class ValidBaselineSubset:
    """Valid baselines subset dataclass. Attributes ending
    on valid are all quantities where at least one baseline
    pair has contributed to the measurement of the source.
    Attributes ending on start are starting points for
    integration windows that end with attributes ending
    on stop.

    Attributes
    ----------
    u_start : :class:`~torch.Tensor`
        Start value for u coverage integration.
    u_stop : :class:`~torch.Tensor`
        Stop value for u coverage integration.
    u_valid : :class:`~torch.Tensor`
        Valid u values.
    v_start : :class:`~torch.Tensor`
        Start value for v coverage integration.
    v_stop : :class:`~torch.Tensor`
        Start value for v coverage integration.
    v_valid : :class:`~torch.Tensor`
        Valid v values.
    w_start : :class:`~torch.Tensor`
        Start value for w coverage integration.
    w_stop : :class:`~torch.Tensor`
        Start value for w coverage integration.
    w_valid : :class:`~torch.Tensor`
        Valid w values.
    baseline_nums : :class:`~torch.Tensor`
        Numbers of baselines per time step.
    date : :class:`~torch.Tensor`
        Time steps of the measurement during which
        at least one baseline pair contributed to the
        measurement.
    q1_start : :class:`~torch.Tensor`
    q1_stop : :class:`~torch.Tensor`
    q1_valid : :class:`~torch.Tensor`
        Valid parallactic angle values (first half of the pair).
    q2_start : :class:`~torch.Tensor`
    q2_stop : :class:`~torch.Tensor`
    q2_valid : :class:`~torch.Tensor`
        Valid parallactic angle values (second half of the pair).
    el1_start : :func:`~torch.tensor`
    el1_stop : :func:`~torch.tensor`
    el1_valid : :func:`~torch.tensor`
        Valid elevation values (first half of the pair).
    el2_start : :func:`~torch.tensor`
    el2_stop : :func:`~torch.tensor`
    el2_valid : :func:`~torch.tensor`
        Valid elevation angle values (second half of the pair).
    st_id_pairs : :class:`~torch.Tensor`
        Station ID pairs for the valid baselines.
    """

    u_start: torch.Tensor
    u_stop: torch.Tensor
    u_valid: torch.Tensor
    v_start: torch.Tensor
    v_stop: torch.Tensor
    v_valid: torch.Tensor
    w_start: torch.Tensor
    w_stop: torch.Tensor
    w_valid: torch.Tensor
    baseline_nums: torch.Tensor
    date: torch.Tensor
    q1_start: torch.Tensor
    q1_stop: torch.Tensor
    q1_valid: torch.Tensor
    q2_start: torch.Tensor
    q2_stop: torch.Tensor
    q2_valid: torch.Tensor
    el1_start: torch.Tensor
    el1_stop: torch.Tensor
    el1_valid: torch.Tensor
    el2_start: torch.Tensor
    el2_stop: torch.Tensor
    el2_valid: torch.Tensor
    st_id_pairs: torch.Tensor

    def __getitem__(self, i):
        """Returns element at index ``i`` for all fields."""
        return ValidBaselineSubset(*[getattr(self, f.name)[i] for f in fields(self)])

    def get_timerange(self, t_start, t_stop):
        """Returns all attributes that fall into the time range
        [``t_start``, ``t_stop``].

        Parameters
        ----------
        t_start : datetime
            Start date.
        t_stop : datetime
            End date.

        Returns
        -------
        ValidBaselineSubset
            :class:`~pyvisgen.simulation.ValidBaselineSubset` dataclass
            object containing all attributes that fall in the time
            range between ``t_start`` and ``t_stop``.
        """
        return ValidBaselineSubset(*[getattr(self, f.name) for f in fields(self)])[
            (self.date >= t_start) & (self.date <= t_stop)
        ]

    def get_unique_grid(
        self,
        fov: float,
        ref_frequency: float,
        img_size: int,
        device: str,
    ):
        """Returns the unique grid for a given FOV, frequency,
        and image size.

        Parameters
        ----------
        fov : float
            Size of the FOV.
        ref_frequency : float
            Reference frequency.
        img_size : int
            Size of the image.
        device : str
            Name of the device to run the operation on,
            e.g. ``'cuda'`` or ``'cpu'``.

        Returns
        -------
        torch.tensor
            Tensor containing the unique grid for a given FOV,
            frequency, and image size.
        """
        uv = torch.cat([self.u_valid[None], self.v_valid[None]], dim=0)

        fov = np.deg2rad(fov / 3600, dtype=np.float128)
        delta = fov ** (-1) * c.value / ref_frequency

        bins = torch.from_numpy(
            np.arange(
                start=-(img_size / 2 + 1 / 2) * delta,
                stop=(img_size / 2 + 1 / 2) * delta,
                step=delta,
                dtype=np.float128,
            ).astype(np.float64)
        ).to(device)

        if len(bins) - 1 > img_size:  # pragma: no cover
            bins = bins[:-1]

        indices_bucket = torch.bucketize(uv, bins)
        indices_bucket_sort = self._lexsort(indices_bucket)
        _, indices_unique_inv, counts = torch.unique_consecutive(
            indices_bucket[:, indices_bucket_sort],
            dim=1,
            return_inverse=True,
            return_counts=True,
        )

        _, ind_sorted = torch.sort(indices_unique_inv, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0], device=device), cum_sum[:-1]))
        first_indices = ind_sorted[cum_sum]

        unique_grid_indices = indices_bucket_sort[first_indices]

        return ValidBaselineSubset(
            *[getattr(self, f.name)[unique_grid_indices] for f in fields(self)]
        )

    def _lexsort(self, a: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Sort a sequence of tensors in lexicographic order.

        Parameters
        ----------
        a : torch.tensor
            Sequence of tensors to sort.
        dim : int, optional
            The dimension along which to sort. Default: ``-1``
        """
        assert dim == -1  # Transpose if you want differently
        assert a.ndim == 2  # Not sure what is numpy behaviour with > 2 dim
        # To be consistent with numpy, we flip the keys (sort by last row first)
        _, inv = torch.unique(a.flip(0), dim=dim, sorted=True, return_inverse=True)

        return torch.argsort(inv)


@dataclass
class Scan:
    """
    Dataclass containing the timing information about a scan.
    A scan is defined as a period in which the simulated interferometer
    measures visibilities. It has a specific start and end time.
    The measurements are taken in regular time steps during the scan time.
    The difference between those steps is the integration time.
    If the total measurement time is not divisible by the integration time
    the last measurement will end prematurely and the time difference between
    the two last measurements will be shorter than the integration time.

    The scan seperation determines the time the interferometer
    does not measure between two scans. The time defined in one scan
    is the time that will be waited after (!) the scan.

    Attributes
    ----------
    start: astropy.times.Time
        The start time of the scan
    stop: astropy.times.Time
        The stop time of the scan
    separation: float
        The separation to the next scan in seconds
    integration_time: float
        The integration time of the interferometer for this scan.
    """

    start: Time
    stop: Time
    separation: float
    integration_time: float

    def __post_init__(self):
        if not isinstance(self.separation, un.Quantity):
            self.separation *= un.second

        if not isinstance(self.integration_time, un.Quantity):
            self.integration_time *= un.second

        if self.separation.unit != un.second:
            self.separation = self.separation.to("second")

        if self.integration_time.unit != un.second:
            self.integration_time = self.integration_time.to("second")

    def get_num_timesteps(self):
        """
        Get the number of time steps in the scan.

        The time steps are spaced by the integration time and the last one is
        the stop time of the scan, so that the whole scan is covered. If the
        duration of the scan is not divisible by the integration time, the
        last interval is shorter than the integration time.

        Returns
        -------

        num_timesteps: int

        """
        duration = (self.stop - self.start).to(un.second)
        # The tolerance absorbs rounding errors in the time difference, e.g.
        # 359.99999999999875 s for a 360 s scan, which would otherwise be
        # treated as a scan that is not covered completely.
        num_intervals = np.ceil((duration / self.integration_time).value - 1e-9)

        return int(num_intervals) + 1

    def get_timesteps(self):
        """
        Get the start and stop times of the timesteps.

        Returns
        -------

        timesteps: astropy.times.Time

        """
        return Time(
            [
                min([self.start + i * self.integration_time, self.stop])
                for i in range(self.get_num_timesteps())
            ]
        )


def create_scans(
    start: Time,
    scan_duration: int | np.typing.ArrayLike,
    num_scans: int,
    scan_separation: int | np.typing.ArrayLike,
    integration_time: int | np.typing.ArrayLike,
) -> list[Scan]:
    """Calculates the individual scans of an observation.

    The first scan starts at ``start``. Every following scan starts after the
    end of the previous scan plus the scan separation.

    Parameters
    ----------
    start : :class:`~astropy.time.Time`
        Start time of the first scan.
    scan_duration : int or ArrayLike
        Scan duration or array of scan durations in seconds
        with size=num_scans.
    num_scans : int
        Number of scans.
    scan_separation : int or ArrayLike
        Scan separation or array of scan separations in seconds
        with size=num_scans - 1.
    integration_time : int or ArrayLike
        Integration time or array of integration times in seconds
        with size=num_scans.

    Returns
    -------
    list[Scan]
        List of scans with a specific start, stop and integration time.
    """
    scans = []

    def _as_array(arr, size: int):
        if np.isscalar(arr):
            return np.ones(size) * arr
        else:
            return np.asarray(arr)

    current_time = start

    scan_duration = _as_array(scan_duration, size=num_scans)
    scan_separation = np.append(_as_array(scan_separation, size=num_scans - 1), 0)
    int_time = _as_array(integration_time, size=num_scans)

    for i in range(num_scans):
        scans.append(
            Scan(
                start=current_time,
                stop=current_time + scan_duration[i] * un.second,
                integration_time=int_time[i] * un.second,
                separation=scan_separation[i] * un.second,
            )
        )

        current_time = scans[-1].stop + scans[-1].separation

    return scans


class Observation:
    """Main observation simulation class.
    The :class:`~pyvisgen.simulation.Observation` class
    simulates the baselines and time steps during the
    observation.

    Parameters
    ----------
    src_ra : float
        Source right ascension coordinate.
    src_dec : float
        Source declination coordinate.
    start_time : datetime
        Observation start time.
    scan_duration : int
        Scan duration.
    num_scans : int
        Number of scans.
    scan_separation : int
        Scan separation.
    integration_time : int
        Integration time.
    ref_frequency : float
        Reference frequency.
    frequency_offsets : list
        Frequency offsets.
    bandwidths : list
        Frequency bandwidth.
    fov : float
        Field of view.
    image_size : int
        Image size of the sky distribution.
    array_layout : str
        Name of an existing array layout. See :mod:`~pyvisgen.layouts`.
    corrupted : bool
        If ``True``, apply corruption during the vis loop.
    device : str
        Torch device to select for computation.
    dense : bool, optional
        If ``True``, apply dense baseline calculation of a perfect
        interferometer. Default: ``False``
    sensitivity_cut : float, optional
        Sensitivity threshold, where only pixels above the value
        are kept. Default: ``1e-6``
    polarization : str, optional
        Choose between ``'linear'`` or ``'circular'`` or ``None`` to
        simulate different types of polarizations or disable
        the simulation of polarization. Default: ``None``
    pol_kwargs : dict, optional
        Additional keyword arguments for the simulation
        of polarization. Default:
        ``{'delta': 0,'amp_ratio': 0.5,'random_state': 42}``
    field_kwargs : dict, optional
        Additional keyword arguments for the random polarization
        field that is applied when simulating polarization.
        Default:
        ``{'order': [1, 1],'scale': [0, 1],'threshold': None,'random_state': 42}``
    show_progress : bool, optional
        If ``True``, show a progress bar during the iteration over the
        scans. Default: ``False``

    Notes
    -----
    See :class:`~pyvisgen.simulation.polarization` and
    :class:`~pyvisgen.simulation.polarization.rand_polarization_field`
    for more information on the keyword arguments in ``pol_kwargs``
    and ``field_kwargs``, respectively.
    """

    def __init__(
        self,
        src_ra: float,
        src_dec: float,
        start_time: datetime,
        scan_duration: int | np.typing.ArrayLike,
        num_scans: int,
        scan_separation: int | np.typing.ArrayLike,
        integration_time: int | np.typing.ArrayLike,
        ref_frequency: float,
        frequency_offsets: list,
        bandwidths: list,
        fov: float,
        image_size: int,
        array_layout: str,
        corrupted: bool,
        device: str,
        dense: bool = False,
        sensitivity_cut: float = 1e-6,
        polarization: str | None = None,
        pol_kwargs: dict = DEFAULT_POL_KWARGS,
        field_kwargs: dict = DEFAULT_FIELD_KWARGS,
        show_progress: bool = False,
    ) -> None:
        """Sets up the observation class.

        Parameters
        ----------
        src_ra : float
            Source right ascension coordinate in degrees.
        src_dec : float
            Source declination coordinate in degrees.
        start_time : datetime
            Observation start time.
        scan_duration : int | ArrayLike
            Scan duration or array of scan durations
            with size=num_scans.
        num_scans : int
            Number of scans.
        scan_separation : int | ArrayLike
            Scan separation or array of scan seperations
            with size=num_scans - 1.
        integration_time : int | ArrayLike
            Integration time or an array of scan seperations
            with size=num_scans.
        ref_frequency : float
            Reference frequency in Hertz.
        frequency_offsets : list
            Frequency offsets in Hertz.
        bandwidths : list
            Frequency bandwidth in Hertz.
        fov : float
            Field of view in arcseconds.
        image_size : int
            Image size of the sky distribution.
        array_layout : str
            Name of an existing array layout. See :mod:`~pyvisgen.layouts`.
        corrupted : bool
            If ``True``, apply corruption during the vis loop.
        device : str
            Torch device to select for computation.
        dense : bool, optional
            If ``True``, apply dense baseline calculation of a perfect
            interferometer. Default: ``False``
        sensitivity_cut : float, optional
            Sensitivity threshold, where only pixels above the value
            are kept. Default: ``1e-6``
        polarization : str, optional
            Choose between ``'linear'`` or ``'circular'`` or ``None`` to
            simulate different types of polarizations or disable
            the simulation of polarization. Default: ``None``
        pol_kwargs : dict, optional
            Additional keyword arguments for the simulation of polarization.
            Default: ``{'delta': 0,'amp_ratio': 0.5,'random_state': 42}``
        field_kwargs : dict, optional
            Additional keyword arguments for the random polarization
            field that is applied when simulating polarization.
            Default:
            ``{'order': [1, 1],'scale': [0, 1],'threshold': None,'random_state': 42}``
        show_progress : bool, optional
            If ``True``, show a progress bar during the iteration over the
            scans. Default: ``False``

        Notes
        -----
        See :class:`~pyvisgen.simulation.polarization` and
        :class:`~pyvisgen.simulation.Polarization.rand_polarization_field`
        for more information on the keyword arguments in ``pol_kwargs``
        and ``field_kwargs``, respectively.
        """
        self.ra = torch.tensor(src_ra).double()
        self.dec = torch.tensor(src_dec).double()

        self.show_progress = show_progress

        self.start = Time(start_time.isoformat(), format="isot", scale="utc")

        self.scan_duration = scan_duration  # Duration of scans (in seconds)
        self.num_scans = num_scans  # Number of scans
        self.scan_separation = scan_separation  # Seperation between scans (in seconds)

        # Integration time (either one time for all scans or one for each scan)
        self.int_time = integration_time

        self.scans = self.create_scans()

        self.ref_frequency = torch.tensor(ref_frequency)
        self.bandwidths = torch.tensor(bandwidths)
        self.frequency_offsets = torch.tensor(frequency_offsets)

        self.waves_low = (
            self.ref_frequency + self.frequency_offsets
        ) - self.bandwidths / 2
        self.waves_high = (
            self.ref_frequency + self.frequency_offsets
        ) + self.bandwidths / 2

        self.fov = fov
        self.img_size = image_size
        self.pix_size = fov / image_size

        self.corrupted = corrupted
        self.sensitivity_cut = sensitivity_cut
        self.device = torch.device(device)

        self.layout = array_layout
        self.array = layouts.get_array_layout(array_layout)
        self.array_earth_loc = EarthLocation.from_geocentric(
            self.array.x, self.array.y, self.array.z, unit=un.m
        )
        self.num_baselines = int(
            len(self.array.st_num) * (len(self.array.st_num) - 1) / 2
        )

        if dense:
            self.waves_low = [self.ref_frequency]
            self.waves_high = [self.ref_frequency]
            self.calc_dense_baselines()
            self.ra = torch.tensor([0]).to(self.device)
            self.dec = torch.tensor([0]).to(self.device)
        else:
            self.calc_baselines()
            self.baselines.num = int(
                self.array.st_num.size(dim=0) * (self.array.st_num.size(dim=0) - 1) / 2
            )
            self.baselines.times_unique = torch.unique(self.baselines.time)

        # lm FIRST: the image grid is defined in the direction cosines and rd is derived
        # from it by inverting the SIN projection (see create_lm_grid).
        self.lm = self.create_lm_grid()
        self.rd = self.create_rd_grid()

        # polarization
        self.polarization = polarization
        self.pol_kwargs = pol_kwargs
        self.field_kwargs = field_kwargs

    def create_scans(self):
        """
        Calculates individual scans of the observation based on
        the number of scans, their duration and the integration time.

        Returns
        -------
        list[Scan]:
            List of scans with a specific start, stop and integration time

        See Also
        --------
        create_scans : Function that is used to calculate the scans.
        """
        return create_scans(
            start=self.start,
            scan_duration=self.scan_duration,
            num_scans=self.num_scans,
            scan_separation=self.scan_separation,
            integration_time=self.int_time,
        )

    def calc_dense_baselines(self):
        """Calculates the baselines of a densely-built
        antenna array, which would provide full coverage of the
        uv space.
        """
        N = self.img_size
        fov = np.deg2rad(self.fov / 3600, dtype=np.float128)
        delta = fov ** (-1) * c.value / self.ref_frequency

        u_dense = torch.from_numpy(
            np.arange(
                start=-(N / 2) * delta,
                stop=(N / 2) * delta,
                step=delta,
                dtype=np.float128,
            ).astype(np.float64)
        ).to(self.device)

        v_dense = u_dense

        uu, vv = torch.meshgrid(u_dense, v_dense, indexing="xy")
        u = uu.flatten()
        v = vv.flatten()

        self.dense_baselines_gpu = ValidBaselineSubset(
            u_start=u,
            u_stop=u,
            u_valid=u,
            v_start=v,
            v_stop=v,
            v_valid=v,
            w_start=torch.zeros(u.shape, device=self.device),
            w_stop=torch.zeros(u.shape, device=self.device),
            w_valid=torch.zeros(u.shape, device=self.device),
            baseline_nums=torch.ones(u.shape, device=self.device),
            date=torch.ones(u.shape, device=self.device),
            q1_start=torch.zeros(u.shape, device=self.device),
            q1_stop=torch.zeros(u.shape, device=self.device),
            q1_valid=torch.zeros(u.shape, device=self.device),
            q2_start=torch.zeros(u.shape, device=self.device),
            q2_stop=torch.zeros(u.shape, device=self.device),
            q2_valid=torch.zeros(u.shape, device=self.device),
            el1_start=torch.zeros(u.shape, device=self.device),
            el1_stop=torch.zeros(u.shape, device=self.device),
            el1_valid=torch.zeros(u.shape, device=self.device),
            el2_start=torch.zeros(u.shape, device=self.device),
            el2_stop=torch.zeros(u.shape, device=self.device),
            el2_valid=torch.zeros(u.shape, device=self.device),
            st_id_pairs=torch.zeros(u.shape, device=self.device),
        )

    def calc_baselines(self):
        """Initializes :class:`~pyvisgen.simulation.Baselines`
        dataclass object and calls
        :py:func:`~pyvisgen.simulation.Observation.get_baselines`
        to compute the contents of the :class:`~pyvisgen.simulation.Baselines`
        dataclass.
        """
        self.baselines = Baselines(
            torch.tensor([]),  # st1
            torch.tensor([]),  # st2
            torch.tensor([]),  # u
            torch.tensor([]),  # v
            torch.tensor([]),  # w
            torch.tensor([]),  # valid
            torch.tensor([]),  # time
            torch.tensor([]),  # q1
            torch.tensor([]),  # q2
            torch.tensor([]),  # el1
            torch.tensor([]),  # el2
        )

        scan_lengths = []

        for scan in tqdm(
            self.scans,
            disable=not self.show_progress,
            desc="Computing baselines",
        ):
            timesteps = scan.get_timesteps()
            bas = self.get_baselines(timesteps)
            self.baselines.add_baseline(bas)
            scan_lengths.append(len(timesteps))

        # Number of time steps per scan. Not a dataclass field, since the fields
        # are concatenated and reshaped, see Baselines.get_valid_subset.
        self.baselines.scan_lengths = scan_lengths

    def get_baselines(self, times):
        """Calculates baselines from source coordinates
        and time of observation for every antenna station
        in array_layout.

        Parameters
        ----------
        times : time object
            time of observation

        Returns
        -------
        dataclass object
            baselines between telescopes with visibility flags
        """
        # catch rare case where dimension of times is 0
        if times.ndim == 0:
            times = Time([times])

        # calculate GHA, local HA, and station elevation for all times.
        GHA, ha_local, el_st_all = self.calc_ref_elev(time=times)

        ar = Array(self.array)
        delta_x, delta_y, delta_z = ar.relative_pos
        st_num_pairs, els_low_pairs, els_high_pairs = ar.antenna_pairs

        baselines = Baselines(
            torch.tensor([]),  # st1
            torch.tensor([]),  # st2
            torch.tensor([]),  # u
            torch.tensor([]),  # v
            torch.tensor([]),  # w
            torch.tensor([]),  # valid
            torch.tensor([]),  # time
            torch.tensor([]),  # q1
            torch.tensor([]),  # q2
            torch.tensor([]),  # el1
            torch.tensor([]),  # el2
        )
        q_all = self.calc_feed_rotation(ha_local)
        q_comb = torch.vstack([torch.combinations(qi) for qi in q_all])
        q_comb = q_comb.reshape(-1, int(q_comb.shape[0] / times.shape[0]), 2)

        # Loop over ha, el_st, times, parallactic angles
        for ha, el_st, time, qc in zip(GHA, el_st_all, times, q_comb):
            u, v, w = self.calc_direction_cosines(ha, el_st, delta_x, delta_y, delta_z)

            # calc current elevations
            cur_el_st = torch.combinations(el_st)

            # calc valid baselines
            m1 = (cur_el_st < els_low_pairs).any(axis=1)
            m2 = (cur_el_st > els_high_pairs).any(axis=1)

            valid = torch.ones(u.shape).bool()
            valid_mask = torch.logical_or(m1, m2)
            valid[valid_mask] = False

            time_mjd = torch.repeat_interleave(
                torch.tensor(time.mjd) * (24 * 60 * 60), len(valid)
            )

            # collect baselines
            base = Baselines(
                st_num_pairs[..., 0],
                st_num_pairs[..., 1],
                u,
                v,
                w,
                valid,
                time_mjd,
                qc[..., 0].ravel(),  # q1
                qc[..., 1].ravel(),  # q2
                cur_el_st[..., 0].ravel(),  # el1
                cur_el_st[..., 1].ravel(),  # el2
            )
            baselines.add_baseline(base)

        return baselines

    def calc_ref_elev(self, time) -> tuple:
        """Calculates the station elevations for given
        time steps.

        Parameters
        ----------
        time : array_like, optional
            Array containing observation time steps.

        Returns
        -------
        tuple
            Tuple containing tensors of the Greenwich hour angle,
            antenna-local hour angles, and the elevations.
        """

        if time.shape == ():
            time = time[None]

        src_crd = SkyCoord(
            ra=self.ra.numpy(), dec=self.dec.numpy(), unit=(un.deg, un.deg)
        )
        # Calculate for all times
        # calculate GHA, Greenwich as reference
        GHA = time.sidereal_time("apparent", "greenwich") - src_crd.ra.to(un.hourangle)

        # calculate local sidereal time and HA at each antenna
        lst = un.Quantity(
            [
                Time(time, location=loc).sidereal_time("mean")
                for loc in self.array_earth_loc
            ]
        )
        ha_local = torch.from_numpy(
            (lst - Longitude(self.ra.item(), unit=un.deg)).radian
        ).T

        # calculate elevations
        el_st_all = src_crd.transform_to(
            AltAz(
                obstime=time[..., None],
                location=EarthLocation.from_geocentric(
                    torch.repeat_interleave(self.array.x[None], len(time), dim=0),
                    torch.repeat_interleave(self.array.y[None], len(time), dim=0),
                    torch.repeat_interleave(self.array.z[None], len(time), dim=0),
                    unit=un.m,
                ),
            )
        )
        if not len(GHA.value) == len(el_st_all):  # pragma: no cover
            raise ValueError(
                "Expected GHA and el_st_all to have the same length"
                f"{len(GHA.value)} and {len(el_st_all)}"
            )

        return (
            torch.tensor(GHA.deg),
            ha_local,
            torch.tensor(el_st_all.alt.degree),
        )

    def calc_feed_rotation(self, ha: torch.Tensor) -> torch.Tensor:
        r"""Calculates feed rotation for every antenna at
        every time step.

        Parameters
        ----------
        ha : torch.tensor
            Local hour angle in radian.

        Returns
        -------
        q : torch.tensor
            Feed rotation, or parallactic angle of the source for
            an observer at :attr:`Observation.array_earth_loc` and
            local hour angle ``ha``.

        Notes
        -----
        The calculation is based on Equation (13.1) of Meeus'
        Astronomical Algorithms:

        .. math::

            q = \atan\left(\frac{\sin h}{\cos\delta
            \tan\varphi - \sin\delta \cos h\right),

        where $h$ is the local hour angle, $\varphi$ the geographical
        latitude of the observer, and $\delta$ the declination of
        the source.
        """
        # We need to create a tensor from the EarthLocation object
        # and save only the geographical latitude of each antenna
        ant_lat = torch.deg2rad(torch.tensor(self.array_earth_loc.lat))
        dec = torch.deg2rad(self.dec)

        # Eqn (13.1) of Meeus' Astronomical Algorithms
        q = torch.arctan2(
            torch.sin(ha),
            torch.tan(ant_lat) * torch.cos(dec) - torch.sin(dec) * torch.cos(ha),
        )

        return q

    def create_lm_grid(self):
        r"""Creates a uniform, square grid in the direction cosines
        :math:`(l, m)` (SIN projection) around the phase centre.

        Returns
        -------
        lm_grid : :func:`torch.tensor`
            Tensor of shape ``(img_size, img_size, 2)``. ``lm_grid[..., 0]``
            contains :math:`l` and ``lm_grid[..., 1]`` contains :math:`m`
            for every pixel. :math:`l` varies along the rows (axis 1),
            :math:`m` along the columns (axis 0). Both are dimensionless
            and have a constant step of ``fov / img_size`` in radians.

        Notes
        -----
        The image grid of an interferometer is uniform and square in
        :math:`(l, m)`. This is the SIN convention encoded by FITS WCS,
        and the frame in which the measurement equation is a plain
        Fourier transform. The RA and Dec values of each pixel,
        :attr:`Observation.rd`, are derived from this grid in
        :meth:`Observation.create_rd_grid`.
        """
        fov = np.deg2rad(self.fov / 3600, dtype=np.float128)
        res = fov / self.img_size
        ax = np.arange(
            start=-(self.img_size / 2) * res,
            stop=(self.img_size / 2) * res,
            step=res,
            dtype=np.float128,
        )[: self.img_size]
        L, M = np.meshgrid(ax, ax, indexing="xy")
        lm_grid = np.stack([L, M], axis=-1).astype(np.float64)
        return torch.from_numpy(lm_grid).to(self.device)

    def create_rd_grid(self):
        r"""Calculates RA and Dec values for every pixel of the grid
        :attr:`Observation.lm` by inverting the SIN projection.

        Returns
        -------
        rd_grid : :func:`torch.tensor`
            Tensor of shape ``(img_size, img_size, 2)``. ``rd_grid[..., 0]``
            contains the RA offset from the phase centre and
            ``rd_grid[..., 1]`` the absolute Dec of each pixel,
            both in radian.

        Notes
        -----
        The pixels are obtained from :attr:`Observation.lm`, which is why
        :meth:`Observation.create_lm_grid` has to be called first. This
        ensures that :attr:`Observation.rd` and :attr:`Observation.lm`
        describe exactly the same pixels. With

        .. math::

            n = \sqrt{1 - l^2 - m^2},

        the inverse of the SIN projection is given by

        .. math::

            \delta = \arcsin\left(m \cos\delta_0
            + n \sin\delta_0\right), \quad
            \Delta\alpha = \arctan2\left(l,
            n \cos\delta_0 - m \sin\delta_0\right),

        where :math:`\delta_0` is the declination of the phase centre.
        """
        d0 = self.dec
        if torch.is_tensor(d0):
            d0 = d0.detach().cpu().numpy()
        dec0 = np.deg2rad(np.asarray(d0, dtype=np.float64).reshape(-1)[0]).astype(
            np.float128
        )

        lm = self.lm.cpu().numpy().astype(np.float128)
        l_coord, m_coord = lm[..., 0], lm[..., 1]
        n = np.sqrt(np.clip(1.0 - l_coord**2 - m_coord**2, 0.0, None))

        # inverse of  l = cos(Dec) sin(dRA),
        # m = sin(Dec) cos(dec0) - cos(Dec) sin(dec0) cos(dRA)
        dec = np.arcsin(m_coord * np.cos(dec0) + n * np.sin(dec0))
        ra = np.arctan2(l_coord, n * np.cos(dec0) - m_coord * np.sin(dec0))

        rd_grid = np.stack([ra, dec], axis=-1).astype(np.float64)
        return torch.from_numpy(rd_grid).to(self.device)

    def calc_direction_cosines(
        self,
        ha: torch.Tensor,
        el_st: torch.Tensor,
        delta_x: torch.Tensor,
        delta_y: torch.Tensor,
        delta_z: torch.Tensor,
    ):
        """Calculates direction cosines u, v, and w for
        given hour angles and relative antenna positions.

        Parameters
        ----------
        ha : :func:`torch.tensor`
            Tensor containing hour angles for each time step.
        el_st : :func:`torch.tensor`
            Tensor containing station elevations.
        delta_x : :func:`torch.tensor`
            Tensor containing relative antenna x-postions.
        delta_y : :func:`torch.tensor`
            Tensor containing relative antenna y-postions.
        delta_z : :func:`torch.tensor`
            Tensor containing relative antenna z-postions.

        Returns
        -------
        u : :func:`torch.tensor`
            Tensor containing direction cosines in u-axis direction.
        v : :func:`torch.tensor`
            Tensor containing direction cosines in v-axis direction.
        w : :func:`torch.tensor`
            Tensor containing direction cosines in w-axis direction.
        """
        src_dec = torch.deg2rad(self.dec)
        ha = torch.deg2rad(ha)
        u = (torch.sin(ha) * delta_x + torch.cos(ha) * delta_y).reshape(-1)
        v = (
            -torch.sin(src_dec) * torch.cos(ha) * delta_x
            + torch.sin(src_dec) * torch.sin(ha) * delta_y
            + torch.cos(src_dec) * delta_z
        ).reshape(-1)
        w = (
            torch.cos(src_dec) * torch.cos(ha) * delta_x
            - torch.cos(src_dec) * torch.sin(ha) * delta_y
            + torch.sin(src_dec) * delta_z
        ).reshape(-1)

        return u, v, w
