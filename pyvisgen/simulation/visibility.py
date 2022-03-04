import numpy as np
from dataclasses import dataclass
import pyvisgen.simulation.utils as ut
import pyvisgen.layouts.layouts as layouts
import astropy.constants as const
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
    num: [int]
    scan: [int]
    base_num: [int]
    u: [float]
    v: [float]
    w: [float]
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
    num: int
    scan: int
    base_num: int
    u: float
    v: float
    w: float
    date: float
    _date: float


def vis_loop(rc, SI, num_threads=48):
    # torch.set_num_threads(num_threads)

    # read config
    array_layout = layouts.get_array_layout(rc["layout"])
    src_crd = SkyCoord(
        ra=170,  # rc["fov_center_ra"].strftime("%H:%M:%S"),
        dec=22,  # rc["fov_center_dec"].strftime("%H:%M:%S"),
        unit=(un.deg, un.deg),
    )
    rc["fov_center_ra"] = 170
    rc["fov_center_dec"] = 22

    wave = np.array(
        [
            const.c / ((rc["base_freq"] + float(freq)) / un.second) / un.meter
            for freq in rc["frequsel"]
        ]
    )

    # calculate rd, lm
    rd = scan.rd_grid(rc["fov_size"] * np.pi / (3600 * 180), rc["img_size"], src_crd)
    lm = scan.lm_grid(rd, src_crd)

    # calculate time steps
    time = ut.calc_time_steps(rc)

    # number statiosn, number baselines
    stat_num = array_layout.st_num.shape[0]
    base_num = int(stat_num * (stat_num - 1) / 2)

    # calculate vis
    visibilities = Visibilities([], [], [], [], [], [], [], [], [], [], [], [])
    vis_num = np.zeros(1)
    # i in total number of scans
    for i in range(rc["scans"]):
        end_idx = int((rc["scan_duration"] / rc["corr_int_time"]) + 1)
        t = time[i * end_idx : (i + 1) * end_idx]

        baselines = scan.get_baselines(src_crd, t, array_layout)

        valid = baselines.valid.reshape(-1, base_num)
        mask = np.array(valid[:-1]).astype(bool) & np.array(valid[1:]).astype(bool)
        u = baselines.u.reshape(-1, base_num)
        v = baselines.v.reshape(-1, base_num)
        w = baselines.w.reshape(-1, base_num)
        base_valid = np.arange(len(baselines.u)).reshape(-1, base_num)[:-1][mask]
        u_valid = u[:-1][mask]
        v_valid = v[:-1][mask]
        w_valid = w[:-1][mask]
        date = np.repeat(
            (t[:-1] + rc["corr_int_time"] * un.second / 2).jd.reshape(-1, 1),
            base_num,
            axis=1,
        )[mask]

        _date = np.zeros(len(u_valid))

        X1 = scan.uncorrupted(lm, baselines, wave[0], t, src_crd, array_layout, SI)
        if X1.shape[0] == 1:
            continue
        X2 = scan.uncorrupted(lm, baselines, wave[0], t, src_crd, array_layout, SI)

        vis_num = np.arange(X1.shape[2] // 2) + 1 + vis_num.max()

        int_values = scan.integrate(X1, X2)
        del X1, X2
        int_values = int_values

        vis = Visibilities(
            int_values[:, 0],
            torch.zeros(int_values.shape[0], dtype=torch.complex128),
            torch.zeros(int_values.shape[0], dtype=torch.complex128),
            torch.zeros(int_values.shape[0], dtype=torch.complex128),
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
    return visibilities
