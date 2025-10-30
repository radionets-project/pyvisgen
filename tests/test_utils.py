def test_read_config():
    from pyvisgen.utils.config import read_data_set_conf

    conf = read_data_set_conf("config/default_data_set.toml")

    assert type(conf) is dict
    assert list(conf.keys()) == [
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
        "corrupted",
        "noisy",
        "sensitivity_cut",
        "polarization",
        "pol_delta",
        "pol_amp_ratio",
        "field_order",
        "field_scale",
        "field_threshold",
        "num_test_images",
        "bundle_size",
        "train_valid_split",
        "grid_size",
        "grid_fov",
        "amp_phase",
        "in_path",
        "out_path_fits",
        "out_path_gridded",
        "dataset_type",
        "gridder",
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
