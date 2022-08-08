import numpy as np
import matplotlib.pyplot as plt
from pyvisgen.simulation.scan import get_baselines
from pyvisgen.layouts.layouts import get_array_layout
from pyvisgen.simulation.utils import read_config, calc_time_steps


def create_sampling_mask(
    layout="vlba",
    size=256,
    base_freq=15.21e10,
    frequsel=[0, 8e7, 1.44e8, 2.08e8],
):
    conf = read_config("../../config/vlba.toml")
    time = calc_time_steps(conf)
    layout = get_array_layout(layout)
    baselines = get_baselines(conf["src_coord"], time, layout)
    u = np.concatenate(
        (baselines.u[baselines.valid == True], -baselines.u[baselines.valid == True])
    )
    v = np.concatenate(
        (baselines.v[baselines.valid == True], -baselines.v[baselines.valid == True])
    )

    u = np.repeat(u[None], len(frequsel), axis=0)
    v = np.repeat(v[None], len(frequsel), axis=0)
    scales = np.array(frequsel) + base_freq
    u /= scales[:, None]
    v /= scales[:, None]

    uv_hist, _, _ = np.histogram2d(
        u.ravel(),
        v.ravel(),
        bins=size,
    )
    mask = uv_hist > 0
    return mask


def sampling(config, sky_dist):
    mask = create_sampling_mask(
        layout=config["layout"],
        size=sky_dist.shape[-1],
        base_freq=config["base_freq"],
        frequsel=config["frequsel"],
    )
    sampled = sky_dist.copy()
    sampled[~mask.astype(bool)] = 0
    plt.imshow(mask[0].astype(bool))
    plt.show()
    return sampled


if __name__ == "__main__":
    sampling()
