from dataclasses import dataclass, fields
from datetime import datetime

import astropy.units as un
import numpy as np
import torch
from astropy.constants import c
from astropy.coordinates import AltAz, Angle, EarthLocation, Longitude, SkyCoord
from astropy.time import Time
from tqdm.auto import tqdm

from pyvisgen.layouts import layouts
from pyvisgen.simulation.array import Array

torch.set_default_dtype(torch.float64)

__all__ = ["Baselines", "ValidBaselineSubset", "Observation"]


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
    st1 : :func:`~torch.tensor`
        Station IDs for antenna pairs.
    st2 : :func:`~torch.tensor`
        Station IDs for antenna pairs.
    u : :func:`~torch.tensor`
        u coordinate coverage.
    v : :func:`~torch.tensor`
        v coordinate coverage.
    w : :func:`~torch.tensor`
        w coordinate coverage.
    valid : :func:`~torch.tensor`
        Mask of valid values, i.e. where the source
        is visible to the antenna pairs.
    time : :func:`~torch.tensor`
        Tensor of observation time steps.
    q1 : :func:`~torch.tensor`
        Tensor of parallactic angle values.
    q2 : :func:`~torch.tensor`
        Tensor of parallactic angle values.
    """

    st1: torch.tensor
    st2: torch.tensor
    u: torch.tensor
    v: torch.tensor
    w: torch.tensor
    valid: torch.tensor
    time: torch.tensor
    q1: torch.tensor
    q2: torch.tensor

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
        ValidBaselineSubset
            :class:`~pyvisgen.simulation.ValidBaselineSubset` dataclass
            object containing valid u, v, and w coverage, observation time
            steps, numbers of baselines, and parallactic angles.
        """
        bas_reshaped = Baselines(
            *[getattr(self, f.name).reshape(-1, num_baselines) for f in fields(self)]
        )

        mask = (bas_reshaped.valid[:-1].bool()) & (bas_reshaped.valid[1:].bool())
        baseline_nums = (
            256 * (bas_reshaped.st1[:-1][mask].ravel() + 1)
            + bas_reshaped.st2[:-1][mask].ravel()
            + 1
        ).to(device)

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
    u_start : :func:`~torch.tensor`
        Start value for u coverage integration.
    u_stop : :func:`~torch.tensor`
        Stop value for u coverage integration.
    u_valid : :func:`~torch.tensor`
        Valid u values.
    v_start : :func:`~torch.tensor`
        Start value for v coverage integration.
    v_stop : :func:`~torch.tensor`
        Start value for v coverage integration.
    v_valid : :func:`~torch.tensor`
        Valid v values.
    w_start : :func:`~torch.tensor`
        Start value for w coverage integration.
    w_stop : :func:`~torch.tensor`
        Start value for w coverage integration.
    w_valid : :func:`~torch.tensor`
        Valid w values.
    baseline_nums : :func:`~torch.tensor`
        Numbers of baselines per time step.
    date : :func:`~torch.tensor`
        Time steps of the measurement during which
        at least one baseline pair contributed to the
        measurement.
    q1_start : :func:`~torch.tensor`
    q1_stop : :func:`~torch.tensor`
    q1_valid : :func:`~torch.tensor`
        Valid parallactic angle values (first half of the pair).
    q2_start : :func:`~torch.tensor`
    q2_stop : :func:`~torch.tensor`
    q2_valid : :func:`~torch.tensor`
        Valid parallactic angle values (second half of the pair).
    """

    u_start: torch.tensor
    u_stop: torch.tensor
    u_valid: torch.tensor
    v_start: torch.tensor
    v_stop: torch.tensor
    v_valid: torch.tensor
    w_start: torch.tensor
    w_stop: torch.tensor
    w_valid: torch.tensor
    baseline_nums: torch.tensor
    date: torch.tensor
    q1_start: torch.tensor
    q1_stop: torch.tensor
    q1_valid: torch.tensor
    q2_start: torch.tensor
    q2_stop: torch.tensor
    q2_valid: torch.tensor

    def __getitem__(self, i):
        """Returns element at index ``i`` for all fields."""
        return torch.stack(
            [
                self.u_start,
                self.u_stop,
                self.u_valid,
                self.v_start,
                self.v_stop,
                self.v_valid,
                self.w_start,
                self.w_stop,
                self.w_valid,
                self.baseline_nums,
                self.date,
                self.q1_start,
                self.q1_stop,
                self.q1_valid,
                self.q2_start,
                self.q2_stop,
                self.q2_valid,
            ]
        )

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
        return ValidBaselineSubset(
            *[getattr(self, f.name).ravel() for f in fields(self)]
        )[(self.date >= t_start) & (self.date <= t_stop)]

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

        if len(bins) - 1 > img_size:
            bins = bins[:-1]

        indices_bucket = torch.bucketize(uv, bins)
        indices_bucket_sort, indices_bucket_inv = self._lexsort(indices_bucket)
        indices_unique, indices_unique_inv, counts = torch.unique_consecutive(
            indices_bucket[:, indices_bucket_sort],
            dim=1,
            return_inverse=True,
            return_counts=True,
        )

        _, ind_sorted = torch.sort(indices_unique_inv, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0], device=device), cum_sum[:-1]))
        first_indices = ind_sorted[cum_sum]

        return self[:][:, indices_bucket_sort[first_indices]]

    def _lexsort(self, a: torch.tensor, dim: int = -1) -> torch.tensor:
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
        a_unq, inv = torch.unique(a.flip(0), dim=dim, sorted=True, return_inverse=True)

        return torch.argsort(inv), inv


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
        scan_duration: int,
        num_scans: int,
        scan_separation: int,
        integration_time: int,
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
        polarization: str = None,
        pol_kwargs: dict = DEFAULT_POL_KWARGS,
        field_kwargs: dict = DEFAULT_FIELD_KWARGS,
        show_progress: bool = False,
    ) -> None:
        """Sets up the observation class.

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

        self.start = Time(start_time.isoformat(), format="isot", scale="utc")
        self.scan_duration = scan_duration
        self.num_scans = num_scans
        self.int_time = integration_time
        self.scan_separation = scan_separation

        self.times, self.times_mjd = self.calc_time_steps()
        self.scans = torch.stack(
            torch.split(
                torch.arange(self.times.size),
                (self.times.size // self.num_scans),
            ),
            dim=0,
        )

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

        self.show_progress = show_progress

        if dense:  # pragma: no cover
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

        self.rd = self.create_rd_grid()
        self.lm = self.create_lm_grid()

        # polarization
        self.polarization = polarization
        self.pol_kwargs = pol_kwargs
        self.field_kwargs = field_kwargs

    def calc_time_steps(self):
        """Computes the time steps of the observation.

        Returns
        -------
        time : array_like
            Array of time steps.
        time.mjd : array_like
            Time steps in mjd format.
        """
        time_lst = [
            self.start
            + self.scan_separation * i * un.second
            + i * self.scan_duration * un.second
            + j * self.int_time * un.second
            for i in range(self.num_scans)
            for j in range(int(self.scan_duration / self.int_time) + 1)
        ]
        # +1 because t_1 is the stop time of t_0.
        # In order to save computing power we take
        # one time more to complete interval
        time = Time(time_lst)

        return time, time.mjd * (60 * 60 * 24)

    def calc_dense_baselines(self):  # pragma: no cover
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

        uu, vv = torch.meshgrid(u_dense, v_dense)
        u = uu.flatten()
        v = vv.flatten()

        self.dense_baselines_gpu = torch.stack(
            [
                u,
                u,
                u,
                v,
                v,
                v,
                torch.zeros(u.shape, device=self.device),
                torch.zeros(u.shape, device=self.device),
                torch.zeros(u.shape, device=self.device),
                torch.ones(u.shape, device=self.device),
                torch.ones(u.shape, device=self.device),
            ]
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
        )

        self.scans = tqdm(
            self.scans,
            disable=not self.show_progress,
            desc="Computing scans",
        )

        for scan in self.scans:
            bas = self.get_baselines(self.times[scan])
            self.baselines.add_baseline(bas)

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
        delta_x, delta_y, delta_z = ar.calc_relative_pos
        st_num_pairs, els_low_pairs, els_high_pairs = ar.calc_ant_pair_vals

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
        )
        q_all = self.calc_feed_rotation(ha_local)
        q_comb = torch.vstack([torch.combinations(qi) for qi in q_all])
        q_comb = q_comb.reshape(-1, int(q_comb.shape[0] / times.shape[0]), 2)

        # Loop over ha, el_st, times, parallactic angles
        for ha, el_st, time, q, qc in zip(GHA, el_st_all, times, q_all, q_comb):
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
                qc[..., 0].ravel(),
                qc[..., 1].ravel(),
            )
            baselines.add_baseline(base)

        return baselines

    def calc_ref_elev(self, time=None) -> tuple:
        """Calculates the station elevations for given
        time steps.

        Parameters
        ----------
        time : array_like or None, optional
            Array containing observation time steps.
            Default: ``None``

        Returns
        -------
        tuple
            Tuple containing tensors of the Greenwich hour angle,
            antenna-local hour angles, and the elevations.
        """
        if time is None:
            time = self.times
        if time.shape == ():
            time = time[None]

        src_crd = SkyCoord(ra=self.ra, dec=self.dec, unit=(un.deg, un.deg))
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
        if not len(GHA.value) == len(el_st_all):
            raise ValueError(
                "Expected GHA and el_st_all to have the same length"
                f"{len(GHA.value)} and {len(el_st_all)}"
            )

        return (
            torch.tensor(GHA.deg),
            ha_local,
            torch.tensor(el_st_all.alt.degree),
        )

    def calc_feed_rotation(self, ha: Angle) -> Angle:
        r"""Calculates feed rotation for every antenna at
        every time step.

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
        ant_lat = torch.tensor(self.array_earth_loc.lat)

        # Eqn (13.1) of Meeus' Astronomical Algorithms
        q = torch.arctan2(
            torch.sin(ha),
            (
                torch.tan(ant_lat) * torch.cos(self.dec)
                - torch.sin(self.dec) * torch.cos(ha)
            ),
        )

        return q

    def create_rd_grid(self):
        """Calculates RA and Dec values for a given fov around a source position

        Parameters
        ----------
        fov : float
            FOV size
        samples : int
            number of pixels
        src_ra :
            right ascensio of the source in deg
        src_dec :
            dec of the source in deg

        Returns
        -------
        rd_grid : 3d array
            Returns a 3d array with every pixel containing a RA and Dec value
        """
        # transform to rad
        fov = np.deg2rad(self.fov / 3600, dtype=np.float128)

        # define resolution
        res = fov / self.img_size

        dec = torch.deg2rad(self.dec).to(self.device)

        r = torch.from_numpy(
            np.arange(
                start=-(self.img_size / 2) * res,
                stop=(self.img_size / 2) * res,
                step=res,
                dtype=np.float128,
            ).astype(np.float64)
        ).to(self.device)
        d = r + dec

        R, _ = torch.meshgrid((r, r), indexing="ij")
        _, D = torch.meshgrid((d, d), indexing="ij")
        rd_grid = torch.cat([R[..., None], D[..., None]], dim=2)

        return rd_grid

    def create_lm_grid(self):
        """Calculates sine projection for fov

        Parameters
        ----------
        rd_grid : 3d array
            array containing a RA and Dec value in every pixel
        src_crd : astropy SkyCoord
            source position

        Returns
        -------
        lm_grid : 3d array
            Returns a 3d array with every pixel containing an l and m value
        """
        dec = np.deg2rad(self.dec.cpu().numpy()).astype(np.float128)

        rd = self.rd.cpu().numpy().astype(np.float128)

        lm_grid = np.zeros(rd.shape, dtype=np.float128)
        lm_grid[..., 0] = np.cos(rd[..., 1]) * np.sin(rd[..., 0])
        lm_grid[..., 1] = np.sin(rd[..., 1]) * np.cos(dec) - np.cos(
            rd[..., 1]
        ) * np.sin(dec) * np.cos(rd[..., 0])

        return torch.from_numpy(lm_grid.astype(np.float64)).to(self.device)

    def calc_direction_cosines(
        self,
        ha: torch.tensor,
        el_st: torch.tensor,
        delta_x: torch.tensor,
        delta_y: torch.tensor,
        delta_z: torch.tensor,
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

        if not (u.shape == v.shape == w.shape):
            raise ValueError(
                "Expected u, v, and w to have the same shapes "
                f"but got {u.shape}, {v.shape}, and {w.shape}."
            )
        return u, v, w
