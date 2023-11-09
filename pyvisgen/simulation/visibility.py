from dataclasses import dataclass

import numpy as np
import torch
from astropy import units as un
from astropy.coordinates import SkyCoord

import pyvisgen.layouts.layouts as layouts
import pyvisgen.simulation.scan as scan
from pyvisgen.simulation.utils import calc_time_steps, calc_valid_baselines, get_IFs
from pyvisgen.simulation.observation import Observation


@dataclass
class Visibilities:
    SI: [complex]
    SQ: [complex]
    SU: [complex]
    SV: [complex]
    num: [float]
    scan: [float]
    base_num: [float]
    u: [un]
    v: [un]
    w: [un]
    date: [float]
    _date: [float]

    def __getitem__(self, i):
        baseline = Vis(
            self.SI[i],
            self.SQ[i],
            self.SU[i],
            self.SV[i],
            self.num[i],
            self.scan[i],
            self.base_num[i],
            self.u[i],
            self.v[i],
            self.w[i],
            self.date[i],
            self._date[i],
        )
        return baseline

    def get_values(self):
        return np.array([self.SI, self.SQ, self.SU, self.SV])

    def add(self, visibilities):
        self.SI = np.concatenate([self.SI, visibilities.SI])
        self.SQ = np.concatenate([self.SQ, visibilities.SQ])
        self.SU = np.concatenate([self.SU, visibilities.SU])
        self.SV = np.concatenate([self.SV, visibilities.SV])
        self.num = np.concatenate([self.num, visibilities.num])
        self.scan = np.concatenate([self.scan, visibilities.scan])
        self.base_num = np.concatenate([self.base_num, visibilities.base_num])
        self.u = np.concatenate([self.u, visibilities.u])
        self.v = np.concatenate([self.v, visibilities.v])
        self.w = np.concatenate([self.w, visibilities.w])
        self.date = np.concatenate([self.date, visibilities.date])
        self._date = np.concatenate([self._date, visibilities._date])


@dataclass
class Vis:
    SI: complex
    SQ: complex
    SU: complex
    SV: complex
    num: float
    scan: float
    base_num: float
    u: un
    v: un
    w: un
    date: float
    _date: float


def vis_loop(rc, SI, num_threads=10, noisy=True):
    torch.set_num_threads(num_threads)

    obs = Observation(
        src_ra=rc["fov_center_ra"],
        src_dec=rc["fov_center_dec"],
        start_time=rc[""],
        scan_duration=rc[""],
        number_scans=rc[""],
        scan_separation=rc[""],
        integration_time=rc[""],
        ref_frequency=rc[""],
        spectral_windows=rc[""],
        bandwiths=rc[""],
        fov=rc[""],
        image_size=rc[""],
        array_layout=rc[""],
    )

    # def number stations and number baselines
    stat_num = len(obs.array.st_num)
    base_num = int(stat_num * (stat_num - 1) / 2)

    # calculate vis
    visibilities = Visibilities(
        np.empty(shape=[0] + [len(IFs)]),
        np.empty(shape=[0] + [len(IFs)]),
        np.empty(shape=[0] + [len(IFs)]),
        np.empty(shape=[0] + [len(IFs)]),
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    vis_num = np.zeros(1)
    for i in range(rc["scans"]):
        end_idx = int((rc["scan_duration"] / rc["corr_int_time"]) + 1)
        t = obs.times[i * end_idx : (i + 1) * end_idx]

        src_crd = SkyCoord(
            ra=self.ra, dec=self.dec, unit=(un.deg, un.deg)
        )

        int_values = []
        for spw in rc["spectral_windows"]:
            val_i = calc_vis(
                obs.lm,
                obs.baselines,
                spw,
                t,
                src_crd,
                obs.array,
                SI,
                obs.rd,
                vis_num,
                corrupted=rc["corrupted"],
            )
            int_values.append(val_i)
            del val_i

        int_values = np.array(int_values)
        if int_values.dtype != np.complex128:
            continue
        int_values = np.swapaxes(int_values, 0, 1)

        if noisy:
            noise = generate_noise(int_values.shape, rc)
            int_values += noise

        vis_num = np.arange(int_values.shape[0]) + 1 + vis_num.max()

        vis = Visibilities(
            torch.tensor(int_values[:, :, 0]),
            torch.zeros(int_values[:, :, 0].shape, dtype=torch.complex128),
            torch.zeros(int_values[:, :, 0].shape, dtype=torch.complex128),
            torch.zeros(int_values[:, :, 0].shape, dtype=torch.complex128),
            vis_num,
            np.repeat(i + 1, len(vis_num)),
            np.array([baselines[i].baselineNum() for i in base_valid]),
            u_valid,
            v_valid,
            w_valid,
            date,
            _date,
        )

        visibilities.add(vis)
        # workaround to guarantee min number of visibilities
        # when num vis is below N sampling is redone
        # if visibilities.get_values().shape[1] < 3500:
        #     return 0
        del int_values
    return visibilities


def calc_vis(
    lm, baselines, wave, t, src_crd, array_layout, SI, rd, vis_num, corrupted=True
):
    if corrupted:
        X1 = scan.direction_independent(
            lm, baselines, wave, t, src_crd, array_layout, SI, rd
        )
        if X1.shape[0] == 1:
            return -1
        X2 = scan.direction_independent(
            lm, baselines, wave, t, src_crd, array_layout, SI, rd
        )
    else:
        X1 = scan.uncorrupted(lm, baselines, wave, t, src_crd, array_layout, SI)
        if X1.shape[0] == 1:
            return -1
        X2 = scan.uncorrupted(lm, baselines, wave, t, src_crd, array_layout, SI)

    int_values = scan.integrate(X1, X2).numpy()
    del X1, X2, SI
    int_values = int_values
    return int_values


def generate_noise(shape, rc):
    # scaling factor for the noise
    factor = 1

    # system efficency factor, near 1
    eta = 0.93

    # taken from simulations
    chan_width = rc["bandwidths"][0] * len(rc["bandwidths"])

    # corr_int_time
    exposure = rc["corr_int_time"]

    # taken from:
    # https://science.nrao.edu/facilities/vla/docs/manuals/oss/performance/sensitivity
    SEFD = 420

    std = factor * 1 / eta * SEFD
    std /= np.sqrt(2 * exposure * chan_width)
    noise = np.random.normal(loc=0, scale=std, size=shape)
    noise = noise + 1.0j * np.random.normal(loc=0, scale=std, size=shape)

    return noise
