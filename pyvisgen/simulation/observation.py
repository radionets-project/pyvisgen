from dataclasses import dataclass
import torch
from astropy.time import Time
import numpy as np
import astropy.units as un
from astropy.coordinates import EarthLocation, AltAz, Angle, SkyCoord
from pyvisgen.layouts import layouts
import astropy.constants as const
from math import pi
from pyvisgen.simulation.utils import Array


@dataclass
class Baselines:
    st1: [object]
    st2: [object]
    u: [float]
    v: [float]
    w: [float]
    valid: [bool]
    time: [float]

    def __getitem__(self, i):
        baseline = Baseline(
            self.st1[i],
            self.st2[i],
            self.u[i],
            self.v[i],
            self.w[i],
            self.valid[i],
            self.time[i],
        )
        return baseline

    def add(self, baselines):
        self.st1 = torch.cat([self.st1, baselines.st1])
        self.st2 = torch.cat([self.st2, baselines.st2])
        self.u = torch.cat([self.u, baselines.u])
        self.v = torch.cat([self.v, baselines.v])
        self.w = torch.cat([self.w, baselines.w])
        self.valid = torch.cat([self.valid, baselines.valid])
        self.time = torch.cat([self.time, baselines.time])

    def baseline_nums(self):
        return 256 * (self.st1 + 1) + self.st2 + 1


@dataclass
class Baseline:
    st1: object
    st2: object
    u: float
    v: float
    w: float
    valid: bool
    time: float

    def baselineNum(self):
        return 256 * (self.st1.st_num + 1) + self.st2.st_num + 1


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
        spectral_windows,
        bandwidths,
        fov,
        image_size,
        array_layout,
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
        # self.frequsel = torch.tensor(frequency_bands)
        self.bandwidths = torch.tensor(bandwidths)
        self.spectral_windows = torch.tensor(spectral_windows)
        self.waves_low = torch.from_numpy(
            (const.c / (self.spectral_windows - self.bandwidths) * un.second / un.meter).value
        )
        self.waves_high = torch.from_numpy(
            (const.c / (self.spectral_windows + self.bandwidths) * un.second / un.meter).value
        )

        self.fov = fov
        self.img_size = image_size
        self.pix_size = fov / image_size

        self.array = layouts.get_array_layout(array_layout)
        self.num_baselines = int(len(self.array.st_num) * (len(self.array.st_num) - 1) / 2)

        self.rd = self.create_rd_grid()
        self.lm = self.create_lm_grid()

        self.calc_baselines()
        self.baselines.num = int(
            len(self.array.st_num) * (len(self.array.st_num) - 1) / 2
        )
        self.baselines.times_unique = torch.unique(self.baselines.time)
        self.calc_valid_baselines()

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
            self.baselines.add(bas)

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
                obstime=time.reshape(len(time), -1),
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
        fov = self.fov * pi / (3600 * 180)

        # define resolution
        res = fov / self.img_size

        ra = torch.deg2rad(self.ra)
        dec = torch.deg2rad(self.dec)
        r = (torch.arange(self.img_size) - self.img_size / 2) * res + ra
        d = -(torch.arange(self.img_size) - self.img_size / 2) * res + dec
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
        lm_grid = torch.zeros(self.rd.shape)
        lm_grid[:, :, 0] = torch.cos(self.rd[:, :, 1]) * torch.sin(
            self.rd[:, :, 0] - torch.deg2rad(self.ra)
        )
        lm_grid[:, :, 1] = torch.sin(self.rd[:, :, 1]) * torch.cos(
            torch.deg2rad(self.dec)
        ) - torch.cos(torch.deg2rad(self.dec)) * torch.sin(
            np.deg2rad(self.dec)
        ) * torch.cos(
            self.rd[:, :, 0] - torch.deg2rad(self.ra)
        )

        return lm_grid

    def get_baselines(self, times):
        """Calculates baselines from source coordinates and time of observation for
        every antenna station in array_layout.

        Parameters
        ----------
        src_crd : astropy SkyCoord object
            ra and dec of source location / pointing center
        time : w time object
            time of observation
        array_layout : dataclass object
            station information

        Returns
        -------
        dataclass object
            baselines between telescopes with visibility flags
        """
        # Calculate for all times
        # calculate GHA, Greenwich as reference
        ha_all, el_st_all = self.calc_ref_elev(time=times)

        ar = Array(self.array)
        delta_x, delta_y, delta_z, indices = ar.calc_relative_pos
        mask = ar.get_baseline_mask
        st_num_pairs, els_low_pairs, els_high_pairs = ar.calc_ant_pair_vals

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
            els_st = self.delete(
                arr=torch.stack(torch.meshgrid(el_st, el_st)).T.reshape(-1, 2),
                ind=mask,
                dim=0,
            )[indices]

            # calc valid baselines
            valid = torch.ones(u.shape).bool()
            m1 = (els_st < els_low_pairs).any(axis=1)
            m2 = (els_st > els_high_pairs).any(axis=1)
            valid_mask = torch.logical_or(m1, m2)
            valid[valid_mask] = False

            time_mjd = torch.repeat_interleave(torch.tensor(time.mjd) * (24 * 60 * 60), len(valid))
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
            baselines.add(base)
        return baselines

    def delete(self, arr: torch.Tensor, ind: int, dim: int) -> torch.Tensor:
        skip = [i for i in range(arr.size(dim)) if i != ind]
        indices = [slice(None) if i != dim else skip for i in range(arr.ndim)]
        return arr.__getitem__(indices)

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

    def calc_valid_baselines(self, time=None):
        if time is None:
            time = self.times
        valid = self.baselines.valid.reshape(-1, self.baselines.num)
        print(valid)
        mask = valid[:-1].bool() & valid[1:].bool()

        self.u_start = self.baselines.u.reshape(-1, self.baselines.num)[:-1][mask]
        self.u_stop = self.baselines.u.reshape(-1, self.baselines.num)[1:][mask]
        self.v_start = self.baselines.v.reshape(-1, self.baselines.num)[:-1][mask]
        self.v_stop = self.baselines.v.reshape(-1, self.baselines.num)[1:][mask]
        self.w_start = self.baselines.w.reshape(-1, self.baselines.num)[:-1][mask]
        self.w_stop = self.baselines.w.reshape(-1, self.baselines.num)[1:][mask]

        self.date = torch.repeat_interleave(
            torch.from_numpy(
                (time[:-1] + self.int_time * un.second / 2).jd.reshape(-1, 1)
            ),
            self.baselines.num,
            dim=1,
        )[mask]
