def test_read_config():
    from pydantic import BaseModel

    from pyvisgen.io import Config

    config = Config.from_toml("config/default_data_set.toml")

    assert issubclass(type(config), BaseModel)

    config_dict = config.to_dict()
    assert list(config_dict.keys()) == ["sampling", "polarization", "bundle"]
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
        "field_threshold"
    ]
    assert list(config_dict["bundle"].keys()) == [
        "dataset_type",
        "in_path",
        "out_path",
        "output_writer",
        "grid_size",
        "grid_fov",
        "amp_phase"
    ]


def test_Array():
    from pyvisgen.layouts.layouts import get_array_layout
    from pyvisgen.simulation.array import Array

    array_layout = get_array_layout("vlba")
    ar = Array(array_layout)
    delta_x, delta_y, delta_z = ar.calc_relative_pos

    st_num_pairs, els_low_pairs, els_high_pairs = ar.calc_ant_pair_vals

    assert delta_x.shape == delta_y.shape == delta_z.shape == (45, 1)
    assert st_num_pairs.shape == els_low_pairs.shape == els_high_pairs.shape == (45, 2)
