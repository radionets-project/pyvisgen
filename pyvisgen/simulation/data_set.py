from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from astropy import units as un
from astropy.coordinates import AltAz, Angle, EarthLocation, SkyCoord
from astropy.time import Time
from tqdm import tqdm

import pyvisgen.fits.writer as writer
import pyvisgen.layouts.layouts as layouts
from pyvisgen.simulation.observation import Observation
from pyvisgen.simulation.visibility import vis_loop
from pyvisgen.utils.config import read_data_set_conf
from pyvisgen.utils.data import load_bundles, open_bundles


def simulate_data_set(config, slurm=False, job_id=None, n=None):
    """
    Wrapper function for simulating visibilities.
    Distinction between slurm and non-threaded simulation.

    Parameters
    ----------
    config : toml file
        path to config file
    slurm : bool
        True, if slurm is used
    job_id : int
        job_id, given by slurm
    n : int
        running index

    """
    conf = read_data_set_conf(config)
    out_path = Path(conf["out_path_fits"])
    out_path.mkdir(parents=True, exist_ok=True)
    data = load_bundles(conf["in_path"])

    if slurm:
        job_id = int(job_id + n * 500)
        out = out_path / Path("vis_" + str(job_id) + ".fits")
        imgs_bundle = len(open_bundles(data[0]))
        bundle = torch.div(job_id, imgs_bundle, rounding_mode="floor")
        image = job_id - bundle * imgs_bundle
        SI = torch.tensor(open_bundles(data[bundle])[image], dtype=torch.cdouble)

        samp_ops = create_sampling_rc(conf)
        vis_data = vis_loop(samp_ops, SI, noisy=conf["noisy"], mode=conf["mode"])

        hdu_list = writer.create_hdu_list(vis_data, samp_ops)
        hdu_list.writeto(out, overwrite=True)

    else:
        for i in range(len(data)):
            SIs = get_images(data, i)

            for j, SI in enumerate(tqdm(SIs)):
                obs, samp_obs = create_observation(conf)
                vis_data = vis_loop(
                    samp_ops, SI, noisy=conf["noisy"], mode=conf["mode"]
                )

                out = out_path / Path("vis_" + str(j) + ".fits")
                hdu_list = writer.create_hdu_list(vis_data, samp_ops)
                hdu_list.writeto(out, overwrite=True)


def get_images(bundles, i):
    SIs = torch.tensor(open_bundles(bundles[i]))
    if len(SIs.shape) == 3:
        SIs = SIs.unsqueeze(1)
    return SIs


def create_observation(conf):
    rc = create_sampling_rc(conf)
    obs = Observation(
        src_ra=rc["fov_center_ra"],
        src_dec=rc["fov_center_dec"],
        start_time=rc["scan_start"],
        scan_duration=rc["scan_duration"],
        num_scans=rc["num_scans"],
        scan_separation=rc["scan_separation"],
        integration_time=rc["corr_int_time"],
        ref_frequency=rc["ref_frequency"],
        frequency_offsets=rc["frequency_offsets"],
        bandwidths=rc["bandwidths"],
        fov=rc["fov_size"],
        image_size=rc["img_size"],
        array_layout=rc["layout"],
        corrupted=rc["corrupted"],
        device=rc["device"],
        sensitivity_cut=rc["sensitivity_cut"],
    )
    return obs, rc


def create_sampling_rc(conf):
    """
    Draw sampling options and test if atleast half of the telescopes can see the source.
    If not, then new parameters are drawn.

    Parameters
    ----------
    conf : dict
        simulation options

    Returns
    -------
    dict
        contains the observation parameters
    """
    samp_ops = draw_sampling_opts(conf)
    array_layout = layouts.get_array_layout(conf["layout"][0])
    half_telescopes = array_layout.x.shape[0] // 2

    while test_opts(samp_ops) <= half_telescopes:
        samp_ops = draw_sampling_opts(conf)

    return samp_ops


def draw_sampling_opts(conf):
    """
    Draw observation options from given intervals.

    Parameters
    ----------
    conf : dict
        simulation options

    Returns
    -------
    dict
        contains randomly drawn observation options
    """
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
    num_scans = np.random.randint(conf["num_scans"][0], conf["num_scans"][1])
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
            num_scans,
            conf["scan_separation"],
            conf["ref_frequency"],
            conf["frequency_offsets"],
            conf["bandwidths"],
            conf["corrupted"],
            conf["device"],
            conf["sensitivty_cut"],
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
        "num_scans": opts[9],
        "scan_separation": opts[10],
        "ref_frequency": opts[11],
        "frequency_offsets": opts[12],
        "bandwidths": opts[13],
        "corrupted": opts[14],
        "device": opts[15],
        "sensitivity_cut": opts[16],
    }
    return samp_ops


def test_opts(rc):
    """
    Compute the number of telescopes that can observe the source given
    certain randomly drawn parameters.

    Parameters
    ----------
    rc : dict
        randomly drawn observational parameters

    Returns
    -------

    """
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


def calc_ref_elev(src_crd, time, array_layout):
    if time.shape == ():
        time = time[None]
    # Calculate for all times
    # calculate GHA, Greenwich as reference for EHT
    ha_all = Angle(
        [t.sidereal_time("apparent", "greenwich") - src_crd.ra for t in time]
    )

    # calculate elevations
    el_st_all = src_crd.transform_to(
        AltAz(
            obstime=time.reshape(len(time), -1),
            location=EarthLocation.from_geocentric(
                np.repeat([array_layout.x], len(time), axis=0),
                np.repeat([array_layout.y], len(time), axis=0),
                np.repeat([array_layout.z], len(time), axis=0),
                unit=un.m,
            ),
        )
    ).alt.degree
    assert len(ha_all.value) == len(el_st_all)
    return ha_all, el_st_all


def calc_time_steps(conf):
    start_time = Time(conf["scan_start"].isoformat(), format="isot")
    scan_separation = conf["scan_separation"]
    num_scans = conf["num_scans"]
    scan_duration = conf["scan_duration"]
    int_time = conf["corr_int_time"]

    time_lst = [
        start_time
        + scan_separation * i * un.second
        + i * scan_duration * un.second
        + j * int_time * un.second
        for i in range(num_scans)
        for j in range(int(scan_duration / int_time) + 1)
    ]
    # +1 because t_1 is the stop time of t_0
    # in order to save computing power we take one time more to complete interval
    time = Time(time_lst)
    return time


if __name__ == "__main__":
    simulate_data_set()
