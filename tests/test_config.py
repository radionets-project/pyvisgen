import pytest
from pydantic import BaseModel, ValidationError

from pyvisgen.io import Config

CONFIG = "tests/test_conf.toml"


def test_read_config():
    config = Config.from_toml(CONFIG)

    assert issubclass(type(config), BaseModel)

    config_dict = config.to_dict()
    assert list(config_dict.keys()) == [
        "sampling",
        "polarization",
        "bundle",
        "datawriter",
        "gridding",
        "fft",
        "codecarbon",
    ]
    assert list(config_dict["sampling"].keys()) == [
        "mode",
        "device",
        "seed",
        "layout",
        "img_size",
        "fov_center_ra",
        "fov_center_dec",
        "fov_size",
        "corr_int_time",
        "scan_start",
        "scan_duration",
        "num_scans",
        "scan_separation",
        "ref_frequency",
        "frequency_offsets",
        "bandwidths",
        "noisy",
        "corrupted",
        "sensitivity_cut",
    ]
    assert list(config_dict["polarization"].keys()) == [
        "mode",
        "delta",
        "amp_ratio",
        "field_order",
        "field_scale",
        "field_threshold",
    ]
    assert list(config_dict["bundle"].keys()) == [
        "dataset_type",
        "in_path",
        "out_path",
        "overlap",
        "grid_size",
        "grid_fov",
        "amp_phase",
    ]

    assert list(config_dict["datawriter"].keys()) == [
        "writer",
        "overlap",
        "shard_pattern",
        "compress",
    ]

    assert list(config_dict["gridding"].keys()) == ["gridder"]
    assert list(config_dict["fft"].keys()) == ["ft"]


def test_unknown_layout():
    config = Config.from_toml(CONFIG)

    with pytest.raises(ValidationError):
        config.sampling.layout = "thislayoutdoesnotexist"


def test_wrong_scan_times():
    config = Config.from_toml(CONFIG)

    with pytest.raises(ValueError):
        config.sampling.scan_start = ["2025-10-28 10:00:00"]


def test_paths():
    config = Config.from_toml(CONFIG)

    with pytest.raises(ValueError):
        config.bundle.in_path = ""

    with pytest.raises(ValueError):
        config.bundle.out_path = ""


def test_writer_selection():
    from pyvisgen.io.datawriters import H5Writer, WDSShardWriter

    config = Config.from_toml(CONFIG)
    assert issubclass(config.datawriter.writer, H5Writer)

    config.datawriter.writer = "wds"
    config = Config.model_validate(config)
    assert issubclass(config.datawriter.writer, WDSShardWriter)
