import numpy as np


def test_read_config():
    from pyvisgen.simulation.utils import read_config

    conf = read_config("config/default.toml")

    assert type(conf) == dict
    assert list(conf.keys()) == [
        "src_coord",
        "fov_size",
        "corr_int_time",
        "scan_start",
        "scan_duration",
        "scans",
        "channel",
        "interval_length",
    ]


def test_single_occurance():
    from pyvisgen.simulation.utils import single_occurance

    arr = np.array([1, 2, 2, 3, 4, 4])

    indx = single_occurance(arr)

    assert (indx == np.array([0, 1, 3, 4])).all()


def test_get_pairs():
    from pyvisgen.layouts.layouts import get_array_layout
    from pyvisgen.simulation.utils import get_pairs

    layout = get_array_layout("eht")
    delta_x, delta_y, delta_z = get_pairs(layout)

    assert delta_x.shape == (56, 1)
    assert delta_x.shape == delta_y.shape == delta_z.shape


def test_calc_time_steps():
    from pyvisgen.simulation.utils import read_config
    from pyvisgen.simulation.utils import calc_time_steps

    conf = read_config("config/default.toml")
    time = calc_time_steps(conf)

    assert time.shape == (2232,)


def test_Array():
    from pyvisgen.simulation.utils import Array
    from pyvisgen.layouts.layouts import get_array_layout

    array_layout = get_array_layout("vlba")
    ar = Array(array_layout)
    delta_x, delta_y, delta_z, indices = ar.calc_relative_pos

    mask = ar.get_baseline_mask

    antenna_pairs, st_num_pairs, els_low_pairs, els_high_pairs = ar.calc_ant_pair_vals

    assert delta_x.shape == delta_y.shape == delta_z.shape == (45, 1)
    assert indices.shape == (45,)
    assert len(mask) == 10
    assert (
        antenna_pairs.shape
        == st_num_pairs.shape
        == els_low_pairs.shape
        == els_high_pairs.shape
        == (45, 2)
    )
