import toml


def read_data_set_conf(conf_toml):
    """Read toml data set configuration file and convert it into a dictionary.

    Parameters
    ----------
    conf : toml file
        path to config file

    Returns
    -------
    sim_conf : dictionary
        simulation configuration
    """
    config = toml.load(conf_toml)
    conf = {}

    conf["mode"] = (config["sampling_options"]["mode"],)
    conf["device"] = config["sampling_options"]["device"]
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
    conf["spectral_windows"] = config["sampling_options"]["spectral_windows"]
    conf["bandwidths"] = config["sampling_options"]["bandwidths"]
    conf["corrupted"] = config["sampling_options"]["corrupted"]
    conf["noisy"] = config["sampling_options"]["noisy"]

    conf["num_test_images"] = config["bundle_options"]["num_test_images"]
    conf["bundle_size"] = config["bundle_options"]["bundle_size"]
    conf["train_valid_split"] = config["bundle_options"]["train_valid_split"]
    conf["grid_size"] = config["bundle_options"]["grid_size"]
    conf["amp_phase"] = config["bundle_options"]["amp_phase"]
    conf["in_path"] = config["bundle_options"]["in_path"]
    conf["out_path_fits"] = config["bundle_options"]["out_path_fits"]
    conf["out_path_gridded"] = config["bundle_options"]["out_path_gridded"]
    return conf
