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
        "sensitivty_cut",
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
        "file_prefix",
    ]


# def test_single_occurance():
#    from pyvisgen.simulation.utils import single_occurance
#
#    arr = np.array([1, 2, 2, 3, 4, 4])
#
#    indx = single_occurance(arr)
#
#    assert (indx == np.array([0, 1, 3, 4])).all()


# def test_get_pairs():
#    from pyvisgen.layouts.layouts import get_array_layout
#    from pyvisgen.simulation.utils import get_pairs
#
#    layout = get_array_layout("eht")
#    delta_x, delta_y, delta_z = get_pairs(layout)
#
#    assert delta_x.shape == (56, 1)
#    assert delta_x.shape == delta_y.shape == delta_z.shape


# def test_calc_time_steps():
#    from pyvisgen.utils.config import read_data_set_conf
#    from pyvisgen.simulation.utils import calc_time_steps
#
#    conf = read_config("config/default_data_set.toml")
#    time = calc_time_steps(conf)
#
#    assert time.shape == (2232,)


def test_Array():
    from pyvisgen.layouts.layouts import get_array_layout
    from pyvisgen.simulation.array import Array

    array_layout = get_array_layout("vlba")
    ar = Array(array_layout)
    delta_x, delta_y, delta_z = ar.calc_relative_pos

    st_num_pairs, els_low_pairs, els_high_pairs = ar.calc_ant_pair_vals

    assert delta_x.shape == delta_y.shape == delta_z.shape == (45, 1)
    assert st_num_pairs.shape == els_low_pairs.shape == els_high_pairs.shape == (45, 2)
