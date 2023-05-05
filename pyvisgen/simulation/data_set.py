from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from astropy import units as un
from astropy.coordinates import SkyCoord
from tqdm import tqdm

import pyvisgen.fits.writer as writer
import pyvisgen.layouts.layouts as layouts
from pyvisgen.simulation.utils import calc_ref_elev, calc_time_steps
from pyvisgen.simulation.visibility import vis_loop
from pyvisgen.utils.config import read_data_set_conf
from pyvisgen.utils.data import load_bundles, open_bundles


def simulate_data_set(config, slurm=False, job_id=None, n=None):
    conf = read_data_set_conf(config)
    out_path = Path(conf["out_path_fits"])
    out_path.mkdir(parents=True, exist_ok=True)

    if slurm:
        job_id = int(job_id + n * 500)
        data = load_bundles(conf["in_path"])
        out = out_path / Path("vis_" + str(job_id) + ".fits")
        imgs_bundle = len(open_bundles(data[0]))
        bundle = torch.div(job_id, imgs_bundle, rounding_mode="floor")
        image = job_id - bundle * imgs_bundle
        SI = torch.tensor(open_bundles(data[bundle])[image], dtype=torch.cdouble)

        samp_ops = create_sampling_rc(conf)
        vis_data = vis_loop(samp_ops, SI, noisy=conf["noisy"])
        while vis_data == 0:
            samp_ops = create_sampling_rc(conf)
            vis_data = vis_loop(samp_ops, SI, noisy=conf["noisy"])
        hdu_list = writer.create_hdu_list(vis_data, samp_ops)
        hdu_list.writeto(out, overwrite=True)

    else:
        data = load_bundles(conf["in_path"])
        for i in range(len(data)):
            SIs = open_bundles(data[i])
            for j, SI in enumerate(tqdm(SIs)):
                out = out_path / Path("vis_" + str(j) + ".fits")
                samp_ops = create_sampling_rc(conf)
                vis_data = vis_loop(samp_ops, SI, noisy=conf["noisy"])
                while vis_data == 0:
                    samp_ops = create_sampling_rc(conf)
                    vis_data = vis_loop(samp_ops, SI, noisy=conf["noisy"])
                hdu_list = writer.create_hdu_list(vis_data, samp_ops)
                hdu_list.writeto(out, overwrite=True)


def create_sampling_rc(conf):
    samp_ops = draw_sampling_opts(conf)
    array_layout = layouts.get_array_layout(conf["layout"][0])
    half_telescopes = array_layout.x.shape[0] // 2

    while test_opts(samp_ops) <= half_telescopes:
        samp_ops = draw_sampling_opts(conf)

    return samp_ops


def draw_sampling_opts(conf):
    angles_ra = np.arange(
        conf["fov_center_ra"][0][0], conf["fov_center_ra"][0][1], step=0.1
    )
    fov_center_ra = np.random.choice(angles_ra)

    angles_dec = np.arange(
        conf["fov_center_dec"][0][0], conf["fov_center_dec"][0][1], step=0.1
    )
    fov_center_dec = np.random.choice(angles_dec)
    start_time_l = datetime.strptime(conf["scan_start"][0], "%d-%m-%Y %H:%M:%S")
    start_time_h = datetime.strptime(conf["scan_start"][1], "%d-%m-%Y %H:%M:%S")
    start_times = pd.date_range(start_time_l, start_time_h, freq="1h").strftime(
        "%d-%m-%Y %H:%M:%S"
    )
    scan_start = np.random.choice(
        [datetime.strptime(time, "%d-%m-%Y %H:%M:%S") for time in start_times]
    )
    scan_duration = np.random.randint(
        conf["scan_duration"][0], conf["scan_duration"][1]
    )
    scans = np.random.randint(conf["scans"][0], conf["scans"][1])
    opts = np.array(
        [
            conf["mode"][0],
            conf["layout"][0],
            conf["img_size"][0],
            fov_center_ra,
            fov_center_dec,
            conf["fov_size"],
            conf["corr_int_time"],
            scan_start,
            scan_duration,
            scans,
            conf["interval_length"],
            conf["base_freq"],
            conf["frequsel"],
            conf["bandwidths"],
            conf["corrupted"],
        ],
        dtype="object",
    )
    samp_ops = {
        "mode": opts[0],
        "layout": opts[1],
        "img_size": opts[2],
        "fov_center_ra": opts[3],
        "fov_center_dec": opts[4],
        "fov_size": opts[5],
        "corr_int_time": opts[6],
        "scan_start": opts[7],
        "scan_duration": opts[8],
        "scans": opts[9],
        "interval_length": opts[10],
        "base_freq": opts[11],
        "frequsel": opts[12],
        "bandwidths": opts[13],
        "corrupted": opts[14],
    }
    return samp_ops


def test_opts(rc):
    array_layout = layouts.get_array_layout(rc["layout"])
    src_crd = SkyCoord(
        ra=rc["fov_center_ra"], dec=rc["fov_center_dec"], unit=(un.deg, un.deg)
    )
    time = calc_time_steps(rc)
    _, el_st_0 = calc_ref_elev(src_crd, time[0], array_layout)
    _, el_st_1 = calc_ref_elev(src_crd, time[1], array_layout)
    el_min = 15
    el_max = 85
    active_telescopes_0 = np.sum((el_st_0 >= el_min) & (el_st_0 <= el_max))
    active_telescopes_1 = np.sum((el_st_1 >= el_min) & (el_st_1 <= el_max))
    return min(active_telescopes_0, active_telescopes_1)


if __name__ == "__main__":
    simulate_data_set()
