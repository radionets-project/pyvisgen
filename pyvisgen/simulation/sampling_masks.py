import re
import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from pyvisgen.simulation.scan import get_baselines
from pyvisgen.layouts.layouts import get_array_layout
from pyvisgen.simulation.utils import read_config, calc_time_steps


def create_sampling_mask(
    layout="vlba",
    size=256,
    multi_channel=True,
    bandwidths=4,
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

    if multi_channel:
        u = np.repeat(u[None], bandwidths, axis=0)
        v = np.repeat(v[None], bandwidths, axis=0)
        scales = np.array(frequsel) + base_freq
        u /= scales[:, None]
        v /= scales[:, None]
    else:
        u /= base_freq
        v /= base_freq
        u = u[None]
        v = v[None]

    uv_hist, _, _ = np.histogram2d(
        u.ravel(),
        v.ravel(),
        bins=size,
    )
    mask = uv_hist > 0
    return mask


def open_sky_distributions(path):
    with h5py.File(path, "r") as hf:
        images = np.array([hf["sky" + str(i)] for i in range(int(len(hf) / 3))])
    return images


def get_data_paths(path):
    data_dir = Path(path)
    bundles = np.array([x for x in data_dir.iterdir()])
    data_paths = np.sort([path for path in bundles if re.findall("source_", path.name)])
    return data_paths


def sampling(path="../../../radiosim/build/example_data"):
    data_paths = get_data_paths(path)
    for data in data_paths:
        images = open_sky_distributions(data)
        mask = np.array([create_sampling_mask() for i in range(len(images))])
        sampled = images.copy()
        sampled[~mask.astype(bool)] = 0
        plt.imshow(mask[0].astype(bool))
        plt.show()


if __name__ == "__main__":
    sampling()
