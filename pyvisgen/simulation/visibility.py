from dataclasses import dataclass

import numpy as np
import torch
from astropy import units as un

import pyvisgen.simulation.scan as scan
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


def vis_loop(rc, SI, num_threads=10, noisy=True):
    torch.set_num_threads(num_threads)

    obs = Observation(
        src_ra=rc["fov_center_ra"],
        src_dec=rc["fov_center_dec"],
        start_time=rc["scan_start"],
        scan_duration=rc["scan_duration"],
        num_scans=rc["num_scans"],
        scan_separation=rc["scan_separation"],
        integration_time=rc["corr_int_time"],
        ref_frequency=rc["ref_frequency"],
        spectral_windows=rc["spectral_windows"],
        bandwidths=rc["bandwidths"],
        fov=rc["fov_size"],
        image_size=rc["img_size"],
        array_layout=rc["layout"],
    )

    # calculate vis
    visibilities = Visibilities(
        np.empty(shape=[0] + [len(obs.spectral_windows)]),
        np.empty(shape=[0] + [len(obs.spectral_windows)]),
        np.empty(shape=[0] + [len(obs.spectral_windows)]),
        np.empty(shape=[0] + [len(obs.spectral_windows)]),
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    vis_num = np.zeros(1)
    for i in range(rc["num_scans"]):
        end_idx = int((rc["scan_duration"] / rc["corr_int_time"]) + 1)
        t = obs.times_mjd[i * end_idx : (i + 1) * end_idx]

        int_values = []
        for spw in rc["spectral_windows"]:
            val_i = calc_vis(
                obs,
                spw,
                t,
                SI,
                vis_num,
                corrupted=rc["corrupted"],
            )
            int_values.append(val_i)
            print(int_values)
            del val_i

        int_values = np.array(int_values)
        if int_values.dtype != np.complex128:
            continue
        int_values = np.swapaxes(int_values, 0, 1)

        if noisy:
            noise = generate_noise(int_values.shape, rc)
            int_values += noise

        vis_num = torch.arange(int_values.shape[0]) + 1 + vis_num.max()

        vis = Visibilities(
            torch.tensor(int_values[:, :, 0]),
            torch.zeros(int_values[:, :, 0].shape, dtype=torch.complex128),
            torch.zeros(int_values[:, :, 0].shape, dtype=torch.complex128),
            torch.zeros(int_values[:, :, 0].shape, dtype=torch.complex128),
            vis_num,
            torch.repeat_interleave(i + 1, len(vis_num)),
            obs.baselines.baseline_nums(),
            obs.baselines.u,
            obs.baselines.v,
            obs.baselines.w,
            obs.baselines.time,
        )

        visibilities.add(vis)
        del int_values
    return visibilities


def calc_vis(obs, spw, t, SI, vis_num, corrupted=True):
    if corrupted:
        X1 = scan.direction_independent(
            obs,
            spw,
            t,
            SI,
        )
        if X1.shape[0] == 1:
            return -1
        X2 = scan.direction_independent(
            obs,
            spw,
            t,
            SI,
        )
    else:
        X1 = scan.uncorrupted(obs, spw, t, SI)
        print("X1", X1)
        if X1.shape[0] == 1:
            return -1
        X2 = scan.uncorrupted(obs, spw, t, SI)
        print("X2", X2)
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
