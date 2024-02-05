from dataclasses import dataclass, fields

import torch
from astropy import units as un

import pyvisgen.simulation.scan as scan


@dataclass
class Visibilities:
    SI: [complex]
    SQ: [complex]
    SU: [complex]
    SV: [complex]
    num: [float]
    # scan: [float]
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


def vis_loop(obs, SI, num_threads=10, noisy=True, mode="full"):
    torch.set_num_threads(num_threads)
    IFs = get_IFs(obs)

    SI = SI.permute(dims=(1, 2, 0)).to(torch.device(obs.device))
    if obs.sensitivity_cut:
        mask = SI > 1e-6
        SI = SI[mask].unsqueeze(-1)
        lm = obs.lm[torch.repeat_interleave(mask, 2, dim=-1)].reshape(-1, 2)
    else:
        lm = obs.lm

    # calculate vis
    visibilities = Visibilities(
        torch.empty(size=[0] + [len(obs.frequency_offsets)]),
        torch.empty(size=[0] + [len(obs.frequency_offsets)]),
        torch.empty(size=[0] + [len(obs.frequency_offsets)]),
        torch.empty(size=[0] + [len(obs.frequency_offsets)]),
        torch.tensor([]),
        torch.tensor([]),
        torch.tensor([]),
        torch.tensor([]),
        torch.tensor([]),
        torch.tensor([]),
    )
    vis_num = torch.zeros(1)
    if mode == "full":
        bas = obs.baselines.get_valid_subset(obs.num_baselines, obs.device)
    if mode == "grid":
        bas = obs.baselines.get_valid_subset(
            obs.num_baselines, obs.device
        ).get_unique_grid(obs.fov, obs.ref_frequency, obs.img_size, obs.device)
    if mode == "dense":
        if obs.device == torch.device("cpu"):
            raise "Only available for GPU calculations!"
        bas = obs.dense_baselines_gpu
        # bas_cpu = obs.dense_baselines_cpu

    spws = [
        calc_windows(torch.tensor(IF), torch.tensor(bandwidth))
        for IF, bandwidth in zip(IFs, obs.bandwidths)
    ]
    print(spws)
    print(obs.waves_low)
    print(obs.waves_high)

    from tqdm import tqdm

    for p in tqdm(torch.arange(bas[:].shape[1]).split(500)):
        bas_p = bas[:][:, p]

        int_values = torch.cat(
            [
                calc_vis(
                    bas_p,
                    lm,
                    wave_low,
                    wave_high,
                    SI,
                    corrupted=obs.corrupted,
                    device=obs.device,
                )[None]
                for wave_low, wave_high in zip(obs.waves_low, obs.waves_high)
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
            int_values[:, :, 0].cpu(),
            torch.zeros(int_values[:, :, 0].shape, dtype=torch.complex128),
            torch.zeros(int_values[:, :, 0].shape, dtype=torch.complex128),
            torch.zeros(int_values[:, :, 0].shape, dtype=torch.complex128),
            vis_num,
            bas_p[9].cpu(),
            bas_p[2].cpu(),
            bas_p[5].cpu(),
            bas_p[8].cpu(),
            bas_p[10].cpu(),
        )

        visibilities.add(vis)
        del int_values
    return visibilities


def calc_vis(bas, lm, spw_low, spw_high, SI, corrupted=False, device="cpu"):
    if corrupted:
        print("Currently not supported!")
        return -1
    else:
        # rime = scan.RIME_uncorrupted(
        #    bas, obs, spw_low, spw_high, device=device, grad=False
        # )
        # int_values = rime(SI)
        int_values = scan.rime(SI, bas, lm, spw_low, spw_high)
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
    IFs = [obs.ref_frequency + float(freq) for freq in obs.frequency_offsets]
    return IFs


def calc_windows(spw, bandwidth):
    return spw - bandwidth * 0.5, spw + bandwidth * 0.5
