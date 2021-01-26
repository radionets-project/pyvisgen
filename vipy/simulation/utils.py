import toml


def read_config(conf):
    config = toml.load(conf)
    sim_conf = {}
    
    sim_conf["fov_center_ra"] = config["sampling_options"]["fov_center_ra"]
    sim_conf["fov_center_dec"] = config["sampling_options"]["fov_center_dec"]
    sim_conf["fov_size"] = config["sampling_options"]["fov_size"]
    sim_conf["corr_int_time"] = config["sampling_options"]["corr_int_time"]
    sim_conf["scan_start"] = config["sampling_options"]["scan_start"]
    sim_conf["scan_duration"] = config["sampling_options"]["scan_duration"]
    sim_conf["scans"] = config["sampling_options"]["scans"]
    sim_conf["channel"] = config["sampling_options"]["channel"]
    sim_conf["interval_length"] = config["sampling_options"]["interval_length"]


    return sim_conf