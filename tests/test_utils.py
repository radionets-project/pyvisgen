def test_read_config():
    from vipy.simulation.utils import read_config

    conf = read_config("config/default.toml")

    assert type(conf) == dict
    assert list(conf.keys()) == [
        "fov_center_ra",
        "fov_center_dec",
        "fov_size",
        "corr_int_time",
        "scan_start",
        "scan_duration",
        "scans",
        "channel",
        "interval_length",
    ]
