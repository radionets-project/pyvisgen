import toml
from astropy.coordinates import SkyCoord
import astropy.units as un
import numpy as np
from astropy.time import Time
from datetime import datetime
import astropy.constants as const
from astropy.coordinates import EarthLocation, AltAz, Angle
from astropy.utils.decorators import lazyproperty


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
    # only calc one half of visibility because of Fourier symmetry
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


class Array:
    def __init__(self, array_layout):
        self.array_layout = array_layout

    @lazyproperty
    def calc_relative_pos(self):
        # from geocentric coordinates to relative coordinates inside array
        delta_x, delta_y, delta_z = get_pairs(self.array_layout)
        self.indices = single_occurance(delta_x)
        delta_x = delta_x[self.indices]
        delta_y = delta_y[self.indices]
        delta_z = delta_z[self.indices]
        return delta_x, delta_y, delta_z, self.indices

    @lazyproperty
    def get_baseline_mask(self):
        # mask baselines between the same telescope
        self.mask = [
            i * len(self.array_layout.x) + i for i in range(len(self.array_layout.x))
        ]
        return self.mask

    @lazyproperty
    def calc_ant_pair_vals(self):
        antenna_pairs = np.delete(
            np.array(
                np.meshgrid(self.array_layout.name, self.array_layout.name)
            ).T.reshape(-1, 2),
            self.mask,
            axis=0,
        )[self.indices]

        st_num_pairs = np.delete(
            np.array(
                np.meshgrid(self.array_layout.st_num, self.array_layout.st_num)
            ).T.reshape(-1, 2),
            self.mask,
            axis=0,
        )[self.indices]

        els_low_pairs = np.delete(
            np.array(
                np.meshgrid(self.array_layout.el_low, self.array_layout.el_low)
            ).T.reshape(-1, 2),
            self.mask,
            axis=0,
        )[self.indices]

        els_high_pairs = np.delete(
            np.array(
                np.meshgrid(self.array_layout.el_high, self.array_layout.el_high)
            ).T.reshape(-1, 2),
            self.mask,
            axis=0,
        )[self.indices]
        return antenna_pairs, st_num_pairs, els_low_pairs, els_high_pairs


def calc_direction_cosines(ha, el_st, delta_x, delta_y, delta_z, src_crd):
    u = (np.sin(ha) * delta_x + np.cos(ha) * delta_y).reshape(-1)
    v = (
        -np.sin(src_crd.ra) * np.cos(ha) * delta_x
        + np.sin(src_crd.ra) * np.sin(ha) * delta_y
        + np.cos(src_crd.ra) * delta_z
    ).reshape(-1)
    w = (
        np.cos(src_crd.ra) * np.cos(ha) * delta_x
        - np.cos(src_crd.ra) * np.sin(ha) * delta_y
        + np.sin(src_crd.ra) * delta_z
    ).reshape(-1)
    assert u.shape == v.shape == w.shape
    return u, v, w


def calc_valid_baselines(baselines, base_num, t, rc):
    valid = baselines.valid.reshape(-1, base_num)
    mask = np.array(valid[:-1]).astype(bool) & np.array(valid[1:]).astype(bool)
    u = baselines.u.reshape(-1, base_num)
    v = baselines.v.reshape(-1, base_num)
    w = baselines.w.reshape(-1, base_num)
    base_valid = np.arange(len(baselines.u)).reshape(-1, base_num)[:-1][mask]
    u_valid = u[:-1][mask]
    v_valid = v[:-1][mask]
    w_valid = w[:-1][mask]
    date = np.repeat(
        (t[:-1] + rc["corr_int_time"] * un.second / 2).jd.reshape(-1, 1),
        base_num,
        axis=1,
    )[mask]

    _date = np.zeros(len(u_valid))
    assert u_valid.shape == v_valid.shape == w_valid.shape
    return base_valid, u_valid, v_valid, w_valid, date, _date
