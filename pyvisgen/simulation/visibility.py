from dataclasses import dataclass

import numpy as np
import torch
from astropy import units as un
from astropy.time import Time

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


def vis_loop(rc, SI, num_threads=10, noisy=True, full=False):
    torch.set_num_threads(num_threads)
    IFs = get_IFs(rc)

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
    vis_num = torch.zeros(1)
    if full:
        bas = obs.baselines.get_valid_subset(obs.num_baselines)
    else:
        bas = obs.baselines.get_valid_subset(obs.num_baselines).get_unique_grid(
            rc["fov_size"], rc["ref_frequency"], rc["img_size"]
        )

    spws = [
        calc_windows(torch.tensor(IF), torch.tensor(bandwidth))
        for IF, bandwidth in zip(IFs, rc["bandwidths"])
    ]

    for i in range(rc["num_scans"]):
        end_idx = int((rc["scan_duration"] / rc["corr_int_time"]) + 1)
        t = obs.times_mjd[i * end_idx : (i + 1) * end_idx]
        for j in range(len(t) - 1):
            t_start = Time(t[j] / (60 * 60 * 24), format="mjd").jd
            t_stop = Time(t[j + 1] / (60 * 60 * 24), format="mjd").jd

            bas_t = bas.get_timerange(t_start, t_stop)

            if bas_t.u_valid.numel() == 0:
                continue

            for p in torch.arange(len(bas_t.u_valid)).split(len(bas_t.u_valid) // 100):
                bas_p = bas_t[p]

                int_values = torch.cat(
                    [
                        calc_vis(
                            bas_p,
                            obs,
                            spw[0],
                            spw[1],
                            SI,
                            corrupted=rc["corrupted"],
                            device=rc["device"],
                        )[None]
                        for spw in spws
                    ]
                )
                if int_values.numel() == 0:
                    continue

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
                    bas_p.baseline_nums,
                    bas_p.u_valid,
                    bas_p.v_valid,
                    bas_p.w_valid,
                    bas_p.date,
                )

                visibilities.add(vis)
                del int_values
    return visibilities


def calc_vis(bas, obs, spw_low, spw_high, SI, corrupted=False, device="cpu"):
    if corrupted:
        print("Currently not supported!")
        return -1
    else:
        rime = scan.RIME_uncorrupted(
            bas, obs, spw_low, spw_high, device=device, grad=False
        )
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


def get_IFs(rc):
    IFs = [rc["ref_frequency"] + float(freq) for freq in rc["spectral_windows"]]
    return IFs


def calc_windows(spw, bandwidth):
    return spw - bandwidth * 0.5, spw + bandwidth * 0.5
