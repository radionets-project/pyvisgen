from dataclasses import dataclass, fields
from math import pi

import astropy.constants as const
import astropy.units as un
import numpy as np
import torch
from astropy.constants import c
from astropy.coordinates import AltAz, Angle, EarthLocation, SkyCoord
from astropy.time import Time

from pyvisgen.layouts import layouts
from pyvisgen.simulation.array import Array


@dataclass
class Baselines:
    st1: torch.tensor
    st2: torch.tensor
    u: torch.tensor
    v: torch.tensor
    w: torch.tensor
    valid: torch.tensor
    time: torch.tensor

    def __getitem__(self, i):
        return Baselines(*[getattr(self, f.name)[i] for f in fields(self)])

    def add_baseline(self, baselines):
        [
            setattr(
                self,
                f.name,
                torch.cat([getattr(self, f.name), getattr(baselines, f.name)]),
            )
            for f in fields(self)
        ]

    def get_valid_subset(self, num_baselines, device):
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

        t = Time(bas_reshaped.time / (60 * 60 * 24), format="mjd").jd
        date = (torch.from_numpy(t[:-1][mask] + t[1:][mask]) / 2).to(device)

        return ValidBaselineSubset(
            baseline_nums,
            u_start,
            u_stop,
            u_valid,
            v_start,
            v_stop,
            v_valid,
            w_start,
            w_stop,
            w_valid,
            date,
        )


@dataclass()
class ValidBaselineSubset:
    baseline_nums: torch.tensor
    u_start: torch.tensor
    u_stop: torch.tensor
    u_valid: torch.tensor
    v_start: torch.tensor
    v_stop: torch.tensor
    v_valid: torch.tensor
    w_start: torch.tensor
    w_stop: torch.tensor
    w_valid: torch.tensor
    date: torch.tensor

    def __getitem__(self, i):
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
            ]
        )

    def get_timerange(self, t_start, t_stop):
        return ValidBaselineSubset(
            *[getattr(self, f.name).ravel() for f in fields(self)]
        )[(self.date >= t_start) & (self.date <= t_stop)]

    def get_unique_grid(self, fov_size, ref_frequency, img_size, device):
        uv = torch.cat([self.u_valid[None], self.v_valid[None]], dim=0)
        fov = fov_size * pi / (3600 * 180)
        delta = 1 / fov * const.c.value.item() / ref_frequency
        bins = (
            torch.arange(
                start=-(img_size / 2) * delta,
                end=(img_size / 2 + 1) * delta,
                step=delta,
                device=device,
            )
            + delta / 2
        )
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

    def _lexsort(self, a, dim=-1):
        assert dim == -1  # Transpose if you want differently
        assert a.ndim == 2  # Not sure what is numpy behaviour with > 2 dim
        # To be consistent with numpy, we flip the keys (sort by last row first)
        a_unq, inv = torch.unique(a.flip(0), dim=dim, sorted=True, return_inverse=True)
        return torch.argsort(inv), inv


class Observation:
    def __init__(
        self,
        src_ra,
        src_dec,
        start_time,
        scan_duration,
        num_scans,
        scan_separation,
        integration_time,
        ref_frequency,
        frequency_offsets,
        bandwidths,
        fov,
        image_size,
        array_layout,
        corrupted,
        device,
        dense=False,
        sensitivity_cut=1e-6,
    ):
        self.ra = torch.tensor(src_ra).float()
        self.dec = torch.tensor(src_dec).float()

        self.start = Time(start_time.isoformat(), format="isot", scale="utc")
        self.scan_duration = scan_duration
        self.num_scans = num_scans
        self.int_time = integration_time
        self.scan_separation = scan_separation
        self.times, self.times_mjd = self.calc_time_steps()
        self.scans = torch.stack(
            torch.split(
                torch.arange(len(self.times)), (len(self.times) // self.num_scans)
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

        self.array = layouts.get_array_layout(array_layout)
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
                len(self.array.st_num) * (len(self.array.st_num) - 1) / 2
            )
            self.baselines.times_unique = torch.unique(self.baselines.time)

        self.rd = self.create_rd_grid()
        self.lm = self.create_lm_grid()

    def calc_dense_baselines(self):
        N = self.img_size
        fov = self.fov * pi / (3600 * 180)
        delta = fov ** (-1) * c.value / self.ref_frequency

        u_dense = torch.arange(
            start=-(N / 2) * delta,
            end=(N / 2) * delta,
            step=delta,
            device=self.device,
            dtype=torch.double,
        )

        v_dense = torch.arange(
            start=-(N / 2) * delta,
            end=(N / 2) * delta,
            step=delta,
            device=self.device,
            dtype=torch.double,
        )

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
        self.baselines = Baselines(
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
        )
        for scan in self.scans:
            bas = self.get_baselines(self.times[scan])
            self.baselines.add_baseline(bas)

    def calc_time_steps(self):
        time_lst = [
            self.start
            + self.scan_separation * i * un.second
            + i * self.scan_duration * un.second
            + j * self.int_time * un.second
            for i in range(self.num_scans)
            for j in range(int(self.scan_duration / self.int_time) + 1)
        ]
        # +1 because t_1 is the stop time of t_0
        # in order to save computing power we take one time more to complete interval
        time = Time(time_lst)
        return time, time.mjd * (60 * 60 * 24)

    def calc_ref_elev(self, time=None):
        if time is None:
            time = self.times
        if time.shape == ():
            time = time[None]
        src_crd = SkyCoord(ra=self.ra, dec=self.dec, unit=(un.deg, un.deg))
        # Calculate for all times
        # calculate GHA, Greenwich as reference
        ha_all = Angle(
            [t.sidereal_time("apparent", "greenwich") - src_crd.ra for t in time]
        )

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
        assert len(ha_all.value) == len(el_st_all)
        return torch.tensor(ha_all.deg), torch.tensor(el_st_all.alt.degree)

    def test_active_telescopes(self):
        _, el_st_0 = self.calc_ref_elev(self.times[0])
        _, el_st_1 = self.calc_ref_elev(self.times[1])
        el_min = 15
        el_max = 85
        active_telescopes_0 = np.sum((el_st_0 >= el_min) & (el_st_0 <= el_max))
        active_telescopes_1 = np.sum((el_st_1 >= el_min) & (el_st_1 <= el_max))
        return min(active_telescopes_0, active_telescopes_1)

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
        3d array
            Returns a 3d array with every pixel containing a RA and Dec value
        """
        # transform to rad
        fov = self.fov / 3600 * (pi / 180)

        # define resolution
        res = fov / self.img_size

        ra = torch.deg2rad(self.ra)
        dec = torch.deg2rad(self.dec)

        r = (torch.arange(self.img_size) - self.img_size / 2) * res
        d = (torch.arange(self.img_size) - self.img_size / 2) * res + dec
        _, R = torch.meshgrid((r, r), indexing="ij")
        D, _ = torch.meshgrid((d, d), indexing="ij")
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
        3d array
            Returns a 3d array with every pixel containing a l and m value
        """
        dec = torch.deg2rad(self.dec)

        lm_grid = torch.zeros(self.rd.shape, device=self.device)
        lm_grid[:, :, 0] = (torch.cos(self.rd[..., 1]) * torch.sin(self.rd[..., 0])).T
        lm_grid[:, :, 1] = (
            torch.sin(self.rd[..., 1]) * torch.cos(dec)
            - torch.cos(self.rd[..., 1]) * torch.sin(dec) * torch.cos(self.rd[..., 0])
        ).T
        return lm_grid

    def get_baselines(self, times):
        """Calculates baselines from source coordinates and time of observation for
        every antenna station in array_layout.

        Parameters
        ----------
        times : time object
            time of observation

        Returns
        -------
        dataclass object
            baselines between telescopes with visibility flags
        """
        # Calculate for all times
        # calculate GHA, Greenwich as reference
        ha_all, el_st_all = self.calc_ref_elev(time=times)

        ar = Array(self.array)
        delta_x, delta_y, delta_z = ar.calc_relative_pos
        st_num_pairs, els_low_pairs, els_high_pairs = ar.calc_ant_pair_vals
        print(els_low_pairs.shape)

        # Loop over ha and el_st
        baselines = Baselines(
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
        )
        for ha, el_st, time in zip(ha_all, el_st_all, times):
            u, v, w = self.calc_direction_cosines(ha, el_st, delta_x, delta_y, delta_z)

            # calc current elevations
            cur_el_st = torch.combinations(el_st)

            # calc valid baselines
            valid = torch.ones(u.shape).bool()
            m1 = (cur_el_st < els_low_pairs).any(axis=1)
            m2 = (cur_el_st > els_high_pairs).any(axis=1)
            valid_mask = torch.logical_or(m1, m2)
            valid[valid_mask] = False

            time_mjd = torch.repeat_interleave(
                torch.tensor(time.mjd) * (24 * 60 * 60), len(valid)
            )
            # collect baselines
            base = Baselines(
                st_num_pairs[:, 0],
                st_num_pairs[:, 1],
                u,
                v,
                w,
                valid,
                time_mjd,
            )
            baselines.add_baseline(base)
        return baselines

    def calc_direction_cosines(self, ha, el_st, delta_x, delta_y, delta_z):
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
        assert u.shape == v.shape == w.shape
        return u, v, w
