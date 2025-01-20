from dataclasses import dataclass, fields

import torch
import toma
from tqdm.autonotebook import tqdm

import pyvisgen.simulation.scan as scan


@dataclass
class Visibilities:
    SI: torch.tensor
    SQ: torch.tensor
    SU: torch.tensor
    SV: torch.tensor
    num: torch.tensor
    base_num: torch.tensor
    u: torch.tensor
    v: torch.tensor
    w: torch.tensor
    date: torch.tensor

    def __getitem__(self, i):
        return Visibilities(*[getattr(self, f.name)[i] for f in fields(self)])

    def get_values(self):
        return torch.cat(
            [self.SI[None], self.SQ[None], self.SU[None], self.SV[None]], dim=0
        ).permute(1, 2, 0)

    def add(self, visibilities):
        [
            setattr(
                self,
                f.name,
                torch.cat([getattr(self, f.name), getattr(visibilities, f.name)]),
            )
            for f in fields(self)
        ]


def vis_loop(
    obs,
    SI,
    num_threads=10,
    noisy=True,
    mode="full",
    batch_size="auto",
    show_progress=False,
):
    torch.set_num_threads(num_threads)
    torch._dynamo.config.suppress_errors = True

    if not (
        isinstance(batch_size, int)
        or (isinstance(batch_size, str) and batch_size == "auto")
    ):
        raise ValueError("Expected batch_size to be 'auto' or of type int")

    SI = torch.flip(SI, dims=[1])

    # define unpolarized sky distribution
    SI = SI.permute(dims=(1, 2, 0))
    I = torch.zeros((SI.shape[0], SI.shape[1], 4), dtype=torch.cdouble)
    I[..., 0] = SI[..., 0]

    # define 2 x 2 Stokes matrix ((I + Q, iU + V), (iU -V, I - Q))
    B = torch.zeros((SI.shape[0], SI.shape[1], 2, 2), dtype=torch.cdouble).to(
        torch.device(obs.device)
    )
    B[:, :, 0, 0] = I[:, :, 0] + I[:, :, 1]
    B[:, :, 0, 1] = I[:, :, 2] + 1j * I[:, :, 3]
    B[:, :, 1, 0] = I[:, :, 2] - 1j * I[:, :, 3]
    B[:, :, 1, 1] = I[:, :, 0] - I[:, :, 1]

    # calculations only for px > sensitivity cut
    mask = (SI >= obs.sensitivity_cut)[..., 0]
    B = B[mask]
    lm = obs.lm[mask]
    rd = obs.rd[mask]

    # normalize visibilities to factor 0.5,
    # so that the Stokes I image is normalized to 1
    B *= 0.5

    # calculate vis
    visibilities = Visibilities(
        torch.empty(size=[0] + [len(obs.waves_low)]),
        torch.empty(size=[0] + [len(obs.waves_low)]),
        torch.empty(size=[0] + [len(obs.waves_low)]),
        torch.empty(size=[0] + [len(obs.waves_low)]),
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
    elif mode == "grid":
        bas = obs.baselines.get_valid_subset(
            obs.num_baselines, obs.device
        ).get_unique_grid(obs.fov, obs.ref_frequency, obs.img_size, obs.device)
    elif mode == "dense":
        if obs.device == torch.device("cpu"):
            raise ValueError("Only available for GPU calculations!")
        obs.calc_dense_baselines()
        bas = obs.dense_baselines_gpu
    else:
        raise ValueError("Unsupported mode!")

    if batch_size == "auto":
        batch_size = bas[:].shape[1]

    visibilities = toma.explicit.batch(
        _batch_loop,
        batch_size,
        visibilities,
        vis_num,
        obs,
        B,
        bas,
        lm,
        rd,
        noisy,
        show_progress,
    )

    return visibilities


def _batch_loop(
    batch_size: int,
    visibilities,
    vis_num: int,
    obs,
    B: torch.tensor,
    bas,
    lm: torch.tensor,
    rd: torch.tensor,
    noisy: bool | float,
    show_progress: bool,
):
    """Main simulation loop of pyvisgen. Computes visibilities
    batchwise.

    Parameters
    ----------
    batch_size : int
        Batch size for loop over Baselines dataclass object.
    visibilities : Visibilities
        Visibilities dataclass object.
    vis_num : int
        Number of visibilities.
    obs : Observation
        Observation class object.
    B : torch.tensor
        Stokes matrix containing stokes visibilities.
    bas : Baselines
        Baselines dataclass object.
    lm : torch.tensor
        lm grid.
    rd : torch.tensor
        rd grid.
    noisy : float or bool
        Simulate noise as SEFD with given value. If set to False,
        no noise is simulated.
    show_progress :
        If True, show a progress bar tracking the loop.

    Returns
    -------
    visibilities : Visibilities
        Visibilities dataclass object.
    """
    batches = torch.arange(bas[:].shape[1]).split(batch_size)
    batches = tqdm(
        batches,
        position=0,
        disable=not show_progress,
        desc="Computing visibilities",
        postfix=f"Batchsize: {batch_size}",
    )

    for p in batches:
        bas_p = bas[:][:, p]

        int_values = torch.cat(
            [
                scan.rime(
                    B,
                    bas_p,
                    lm,
                    rd,
                    obs.ra,
                    obs.dec,
                    torch.unique(obs.array.diam),
                    wave_low,
                    wave_high,
                    corrupted=obs.corrupted,
                )[None]
                for wave_low, wave_high in zip(obs.waves_low, obs.waves_high)
            ]
        )
        if int_values.numel() == 0:
            continue

        int_values = torch.swapaxes(int_values, 0, 1)

        if noisy != 0:
            noise = generate_noise(int_values.shape, obs, noisy)
            int_values += noise

        vis_num = torch.arange(int_values.shape[0]) + 1 + vis_num.max()

        vis = Visibilities(
            int_values[:, :, 0, 0].cpu(),
            int_values[:, :, 0, 1].cpu(),
            int_values[:, :, 1, 0].cpu(),
            int_values[:, :, 1, 1].cpu(),
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


def generate_noise(shape, obs, SEFD):
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

    std = factor * 1 / eta * SEFD
    std /= torch.sqrt(2 * exposure * chan_width)
    noise = torch.normal(mean=0, std=std, size=shape, device=obs.device)
    noise = noise + 1.0j * torch.normal(mean=0, std=std, size=shape, device=obs.device)

    return noise
