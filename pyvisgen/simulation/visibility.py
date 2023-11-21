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

        # get baseline subset
        bas_t = obs.baselines[
            (obs.baselines.time >= t[0]) & (obs.baselines.time <= t[-1])
        ]
        bas_t.calc_valid_baselines(obs.num_baselines)
        if bas_t.valid.numel() == 0:
            continue

        int_values = torch.cat(
            [
                calc_vis(
                    bas_t,
                    obs,
                    spw,
                    SI,
                    corrupted=rc["corrupted"],
                    device=rc["device"],
                )[None]
                for spw in rc["spectral_windows"]
            ]
        )

        int_values = torch.swapaxes(int_values, 0, 1)

        if noisy:
            noise = generate_noise(int_values.shape, rc)
            int_values += noise

        vis_num = torch.arange(int_values.shape[0]) + 1 + vis_num.max()

        vis = Visibilities(
            int_values[:, :, 0],
            torch.zeros(int_values[:, :, 0].shape, dtype=torch.complex128),
            torch.zeros(int_values[:, :, 0].shape, dtype=torch.complex128),
            torch.zeros(int_values[:, :, 0].shape, dtype=torch.complex128),
            vis_num,
            torch.repeat_interleave(torch.tensor(i) + 1, len(vis_num)),
            bas_t.baseline_nums,
            bas_t.u_valid,
            bas_t.v_valid,
            bas_t.w_valid,
            bas_t.date,
        )

        visibilities.add(vis)
        del int_values
    return visibilities


def calc_vis(bas, obs, spw, SI, corrupted=False, device="cpu"):
    if corrupted:
        print("Currently not supported!")
        return -1
    else:
        rime = scan.RIME_uncorrupted(bas, obs, spw, device=device, grad=False)
        int_values = rime(SI.permute(dims=(1, 2, 0)).to(torch.device(device)))
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
    std /= torch.sqrt(2 * exposure * chan_width)
    noise = torch.normal(mean=0, std=std, size=shape)
    noise = noise + 1.0j * torch.normal(mean=0, std=std, size=shape)

    return noise
