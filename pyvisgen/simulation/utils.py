import toml
from astropy.coordinates import SkyCoord
import astropy.units as un
import numpy as np
from astropy.time import Time
from datetime import datetime
import astropy.constants as const
from astropy.coordinates import EarthLocation, AltAz, Angle


def read_config(conf):
    """Read toml simulation configuration file and convert it into a dictionary.

    Parameters
    ----------
    conf : toml file
        path to config file

    Returns
    -------
    sim_conf : dictionary
        simulation configuration
    """
    config = toml.load(conf)
    sim_conf = {}

    sim_conf["src_coord"] = SkyCoord(
        ra=config["sampling_options"]["fov_center_ra"],
        dec=config["sampling_options"]["fov_center_dec"],
        unit=(un.deg, un.deg),
    )
    sim_conf["fov_size"] = config["sampling_options"]["fov_size"]
    sim_conf["corr_int_time"] = config["sampling_options"]["corr_int_time"]
    sim_conf["scan_start"] = datetime.strptime(
        config["sampling_options"]["scan_start"], "%d-%m-%Y %H:%M:%S"
    )
    sim_conf["scan_duration"] = config["sampling_options"]["scan_duration"]
    sim_conf["scans"] = config["sampling_options"]["scans"]
    sim_conf["channel"] = config["sampling_options"]["channel"]
    sim_conf["interval_length"] = config["sampling_options"]["interval_length"]
    return sim_conf


def single_occurance(array):
    vals, index = np.unique(np.abs(array), return_index=True)
    return index


def get_pairs(array_layout):
    delta_x = (
        np.array(
            [val - array_layout.x[val - array_layout.x != 0] for val in array_layout.x]
        )
        .ravel()
        .reshape(-1, 1)
    )
    delta_y = (
        np.array(
            [val - array_layout.y[val - array_layout.y != 0] for val in array_layout.y]
        )
        .ravel()
        .reshape(-1, 1)
    )
    delta_z = (
        np.array(
            [val - array_layout.z[val - array_layout.z != 0] for val in array_layout.z]
        )
        .ravel()
        .reshape(-1, 1)
    )
    return delta_x, delta_y, delta_z


def calc_time_steps(conf):
    start_time = Time(conf["scan_start"].isoformat(), format="isot")
    interval = conf["interval_length"]
    num_scans = conf["scans"]
    scan_duration = conf["scan_duration"]
    int_time = conf["corr_int_time"]

    time_lst = [
        start_time + interval * i * un.second + j * int_time * un.second
        for i in range(num_scans)
        for j in range(int(scan_duration / int_time) + 1)
    ]
    # +1 because t_1 is the stop time of t_0
    # in order to save computing power we take one time more to complete interval
    time = Time(time_lst)

    return time


def get_IFs(rc):
    IFs = np.array(
        [
            const.c / ((rc["base_freq"] + float(freq)) / un.second) / un.meter
            for freq in rc["frequsel"]
        ]
    )
    return IFs


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
