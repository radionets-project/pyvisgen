from dataclasses import dataclass, fields

import torch
from astropy import units as un
from astropy.time import Time

import pyvisgen.simulation.scan as scan


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
        return Visibilities(*[getattr(self, f.name)[i] for f in fields(self)])

    def get_values(self):
        return torch.cat(
            [self.SI[None], self.SQ[None], self.SU[None], self.SV[None]], dim=0
        )

    def add(self, visibilities):
        [
            setattr(
                self,
                f.name,
                torch.cat([getattr(self, f.name), getattr(visibilities, f.name)]),
            )
            for f in fields(self)
        ]


def vis_loop(obs, SI, num_threads=10, noisy=True, full=False):
    torch.set_num_threads(num_threads)
    IFs = get_IFs(obs)

    # calculate vis
    visibilities = Visibilities(
        torch.empty(size=[0] + [len(obs.spectral_windows)]),
        torch.empty(size=[0] + [len(obs.spectral_windows)]),
        torch.empty(size=[0] + [len(obs.spectral_windows)]),
        torch.empty(size=[0] + [len(obs.spectral_windows)]),
        torch.tensor([]),
        torch.tensor([]),
        torch.tensor([]),
        torch.tensor([]),
        torch.tensor([]),
        torch.tensor([]),
        torch.tensor([]),
    )
    vis_num = torch.zeros(1)
    if full:
        bas = obs.baselines.get_valid_subset(obs.num_baselines)
    else:
        bas = obs.baselines.get_valid_subset(obs.num_baselines).get_unique_grid(
            obs.fov, obs.ref_frequency, obs.img_size
        )

    spws = [
        calc_windows(torch.tensor(IF), torch.tensor(bandwidth))
        for IF, bandwidth in zip(IFs, obs.bandwidths)
    ]

    for i in range(obs.num_scans):
        end_idx = int((obs.scan_duration / obs.int_time) + 1)
        t = obs.times_mjd[i * end_idx : (i + 1) * end_idx]
        for j in range(len(t) - 1):
            t_start = Time(t[j] / (60 * 60 * 24), format="mjd").jd
            t_stop = Time(t[j + 1] / (60 * 60 * 24), format="mjd").jd

            bas_t = bas.get_timerange(t_start, t_stop)

            if bas_t.u_valid.numel() == 0:
                continue

            for p in torch.arange(len(bas_t.u_valid)).split(1):
                bas_p = bas_t[p]

                int_values = torch.cat(
                    [
                        calc_vis(
                            bas_p,
                            obs,
                            spw[0],
                            spw[1],
                            SI,
                            corrupted=obs.corrupted,
                            device=obs.device,
                        )[None]
                        for spw in spws
                    ]
                )
                if int_values.numel() == 0:
                    continue

                int_values = torch.swapaxes(int_values, 0, 1)

                if noisy:
                    noise = generate_noise(int_values.shape, obs)
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


def generate_noise(shape, obs):
    # scaling factor for the noise
    factor = 1

    # system efficency factor, near 1
    eta = 0.93

    # taken from simulations
    chan_width = obs.bandwidths[0] * len(obs.bandwidths)

    # corr_int_time
    exposure = obs.int_time

    # taken from:
    # https://science.nrao.edu/facilities/vla/docs/manuals/oss/performance/sensitivity
    SEFD = 420

    std = factor * 1 / eta * SEFD
    std /= torch.sqrt(2 * exposure * chan_width)
    noise = torch.normal(mean=0, std=std, size=shape)
    noise = noise + 1.0j * torch.normal(mean=0, std=std, size=shape)

    return noise


def get_IFs(obs):
    IFs = [obs.ref_frequency + float(freq) for freq in obs.spectral_windows]
    return IFs


def calc_windows(spw, bandwidth):
    return spw - bandwidth * 0.5, spw + bandwidth * 0.5
