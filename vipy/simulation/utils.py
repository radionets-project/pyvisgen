import toml
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np


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
        unit=(u.hourangle, u.deg),
    )
    sim_conf["fov_size"] = config["sampling_options"]["fov_size"]
    sim_conf["corr_int_time"] = config["sampling_options"]["corr_int_time"]
    sim_conf["scan_start"] = config["sampling_options"]["scan_start"]
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
