def test_Array():
    from pyvisgen.layouts.layouts import get_array_layout
    from pyvisgen.simulation.array import Array

    array_layout = get_array_layout("vlba")
    ar = Array(array_layout)
    delta_x, delta_y, delta_z = ar.calc_relative_pos

    st_num_pairs, els_low_pairs, els_high_pairs = ar.calc_ant_pair_vals

    assert delta_x.shape == delta_y.shape == delta_z.shape == (45, 1)
    assert st_num_pairs.shape == els_low_pairs.shape == els_high_pairs.shape == (45, 2)
