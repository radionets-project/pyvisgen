import numpy as np
from dataclasses import dataclass
from pyvisgen.simulation.utils import get_IFs, calc_valid_baselines, calc_time_steps
import pyvisgen.layouts.layouts as layouts
from astropy import units as un
import pyvisgen.simulation.scan as scan
import torch
from astropy.coordinates import SkyCoord


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


def vis_loop(rc, SI):
    # define array, source coords, and IFs
    array_layout = layouts.get_array_layout(rc["layout"])
    src_crd = SkyCoord(
        ra=rc["fov_center_ra"], dec=rc["fov_center_dec"], unit=(un.deg, un.deg),
    )

    # define IFs
    IFs = get_IFs(rc)

    # calculate time steps
    time = calc_time_steps(rc)

    # calculate rd, lm
    rd = scan.rd_grid(rc["fov_size"], rc["img_size"], src_crd)
    lm = scan.lm_grid(rd, src_crd)

    # def number stations and number baselines
    stat_num = array_layout.st_num.shape[0]
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
        t = time[i * end_idx : (i + 1) * end_idx]

        baselines = scan.get_baselines(src_crd, t, array_layout)

        base_valid, u_valid, v_valid, w_valid, date, _date = calc_valid_baselines(
            baselines, base_num, t, rc
        )

        int_values = np.array(
            [
                calc_vis(lm, baselines, IF, t, src_crd, array_layout, SI, rd, vis_num,)
                for IF in IFs
            ]
        )
        if int_values.dtype != np.complex128:
            continue
        int_values = np.swapaxes(int_values, 0, 1)
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
    return visibilities


def calc_vis(lm, baselines, wave, t, src_crd, array_layout, SI, rd, vis_num):
    X1 = scan.uncorrupted(lm, baselines, wave, t, src_crd, array_layout, SI)  # , rd
    if X1.shape[0] == 1:
        return -1
    X2 = scan.uncorrupted(lm, baselines, wave, t, src_crd, array_layout, SI)  # , rd

    int_values = scan.integrate(X1, X2).numpy()
    del X1, X2
    int_values = int_values
    return int_values
