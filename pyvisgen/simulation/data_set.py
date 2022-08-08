import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from pyvisgen.utils.config import read_data_set_conf
from pyvisgen.simulation.visibility import vis_loop
import pyvisgen.fits.writer as writer
from radiosim.data import radiosim_data


def simulate_data_set(config, slurm=False, job_id=None, n=None):
    np.random.seed(1)
    conf = read_data_set_conf(config)
    out_path = Path(conf["out_path"])
    out_path.mkdir(parents=True, exist_ok=True)

    if slurm:
        job_id = int(job_id + n * 1000)
        data = radiosim_data(conf["in_path"])
        out = out_path / Path("vis_" + str(job_id) + ".fits")
        SI = torch.tensor(data[job_id][0][0], dtype=torch.cdouble)

        samp_ops = create_sampling_rc(conf)
        vis_data = vis_loop(samp_ops, SI)
        while vis_data == 0:
            samp_ops = create_sampling_rc(conf)
            vis_data = vis_loop(samp_ops, SI)
        hdu_list = writer.create_hdu_list(vis_data, samp_ops)
        hdu_list.writeto(out, overwrite=True)

    else:
        data = radiosim_data(conf["in_path"])
        for i in tqdm(range(len(data))):
            out = out_path / Path("vis_" + str(i) + ".fits")
            SI = torch.tensor(data[i][0][0], dtype=torch.cdouble)

            samp_ops = create_sampling_rc(conf)
            vis_data = vis_loop(samp_ops, SI)
            while vis_data == 0:
                samp_ops = create_sampling_rc(conf)
                vis_data = vis_loop(samp_ops, SI)
            hdu_list = writer.create_hdu_list(vis_data, samp_ops)
            hdu_list.writeto(out, overwrite=True)


def create_sampling_rc(conf):
    sampling_opts = draw_sampling_opts(conf)
    samp_ops = {
        "mode": sampling_opts[0],
        "layout": sampling_opts[1],
        "img_size": sampling_opts[2],
        "fov_center_ra": sampling_opts[3],
        "fov_center_dec": sampling_opts[4],
        "fov_size": sampling_opts[5],
        "corr_int_time": sampling_opts[6],
        "scan_start": sampling_opts[7],
        "scan_duration": sampling_opts[8],
        "scans": sampling_opts[9],
        "interval_length": sampling_opts[10],
        "base_freq": sampling_opts[11],
        "frequsel": sampling_opts[12],
        "bandwidths": sampling_opts[13],
    }
    return samp_ops


def draw_sampling_opts(conf):
    date_str_ra = pd.date_range(
        conf["fov_center_ra"][0][0].strftime("%H:%M:%S"),
        conf["fov_center_ra"][0][1].strftime("%H:%M:%S"),
        freq="1min",
    ).strftime("%H:%M:%S")
    times_ra = [
        datetime.time(datetime.strptime(date, "%H:%M:%S")) for date in date_str_ra
    ]
    fov_center_ra = np.random.choice(times_ra)
    angles_dec = np.arange(
        conf["fov_center_dec"][0][0],
        conf["fov_center_dec"][0][1],
        step=0.1,
    )
    fov_center_dec = np.random.choice(angles_dec)
    start_time_l = datetime.strptime(conf["scan_start"][0], "%d-%m-%Y %H:%M:%S")
    start_time_h = datetime.strptime(conf["scan_start"][1], "%d-%m-%Y %H:%M:%S")
    start_times = pd.date_range(
        start_time_l,
        start_time_h,
        freq="1h",
    ).strftime("%d-%m-%Y %H:%M:%S")
    scan_start = np.random.choice(
        [datetime.strptime(time, "%d-%m-%Y %H:%M:%S") for time in start_times]
    )
    scan_duration = (
        np.random.randint(conf["scan_duration"][0], conf["scan_duration"][1])
        * conf["corr_int_time"]
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
        ],
        dtype="object",
    )
    return opts


if __name__ == "__main__":
    simulate_data_set("/net/big-tank/POOL/projects/radio/test_rime/create_dataset.toml")
