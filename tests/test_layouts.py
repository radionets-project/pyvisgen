import numpy as np


def test_get_array_layout():
    from vipy.layouts.layouts import get_array_layout

    layout = get_array_layout("eht")

    assert len(layout.name) == 8
    assert type(layout[0].name) == str
    assert type(layout[0].x) == np.float64
    assert type(layout[0].y) == np.float64
    assert type(layout[0].z) == np.float64
    assert type(layout[0].diam) == np.float64
    assert type(layout[0].el_low) == np.int64
    assert type(layout[0].el_high) == np.int64
    assert type(layout[0].sefd) == np.int64
    assert type(layout[0].altitude) == np.int64
