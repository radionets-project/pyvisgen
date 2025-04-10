from pathlib import Path

import toml


def read_data_set_conf(conf_toml: str | Path) -> dict:
    """Read toml data set configuration file and convert
    it into a dictionary.

    Parameters
    ----------
    conf_toml : str or Path
        Path to config file.

    Returns
    -------
    conf : dict
        Simulation configuration.
    """
    config = toml.load(conf_toml)
    config = sanitize_conf(config)
    conf = {}

    conf["mode"] = config["sampling_options"]["mode"]
    conf["device"] = config["sampling_options"]["device"]
    conf["seed"] = config["sampling_options"]["seed"]
    conf["layout"] = (config["sampling_options"]["layout"],)
    conf["img_size"] = (config["sampling_options"]["img_size"],)
    conf["fov_center_ra"] = (config["sampling_options"]["fov_center_ra"],)
    conf["fov_center_dec"] = (config["sampling_options"]["fov_center_dec"],)
    conf["fov_size"] = config["sampling_options"]["fov_size"]
    conf["corr_int_time"] = config["sampling_options"]["corr_int_time"]
    conf["scan_start"] = config["sampling_options"]["scan_start"]
    conf["scan_duration"] = config["sampling_options"]["scan_duration"]
    conf["num_scans"] = config["sampling_options"]["num_scans"]
    conf["scan_separation"] = config["sampling_options"]["scan_separation"]
    conf["ref_frequency"] = config["sampling_options"]["ref_frequency"]
    conf["frequency_offsets"] = config["sampling_options"]["frequency_offsets"]
    conf["bandwidths"] = config["sampling_options"]["bandwidths"]
    conf["corrupted"] = config["sampling_options"]["corrupted"]
    conf["noisy"] = config["sampling_options"]["noisy"]
    conf["sensitivty_cut"] = config["sampling_options"]["sensitivity_cut"]

    conf["polarization"] = config["polarization_options"]["mode"]
    conf["pol_delta"] = config["polarization_options"]["delta"]
    conf["pol_amp_ratio"] = config["polarization_options"]["amp_ratio"]
    conf["field_order"] = config["polarization_options"]["field_order"]
    conf["field_scale"] = config["polarization_options"]["field_scale"]
    conf["field_threshold"] = config["polarization_options"]["field_threshold"]

    conf["num_test_images"] = config["bundle_options"]["num_test_images"]
    conf["bundle_size"] = config["bundle_options"]["bundle_size"]
    conf["train_valid_split"] = config["bundle_options"]["train_valid_split"]
    conf["grid_size"] = config["bundle_options"]["grid_size"]
    conf["grid_fov"] = config["bundle_options"]["grid_fov"]
    conf["amp_phase"] = config["bundle_options"]["amp_phase"]
    conf["in_path"] = config["bundle_options"]["in_path"]
    conf["out_path_fits"] = config["bundle_options"]["out_path_fits"]
    conf["out_path_gridded"] = config["bundle_options"]["out_path_gridded"]
    conf["file_prefix"] = config["bundle_options"]["file_prefix"]

    # handle case if file_prefix = None
    if not conf["file_prefix"]:
        conf["file_prefix"] = ""

    return conf


def sanitize_conf(conf: dict) -> dict:
    """Sanitizes a given dict by replacinginstances of
    'none' str with None.

    Parameters
    ----------
    conf : list
        Unsanitized config dict.

    Returns
    -------
    sanitized_conf : list
        Sanitized conf dict where all instances of 'none'
        are replaced with None.
    """
    sanitized_conf = {}
    for key, val in conf.items():
        if isinstance(val, dict):
            val = sanitize_conf(val)
        elif isinstance(val, str) and val == "none":
            val = None

        sanitized_conf[key] = val

    return sanitized_conf
