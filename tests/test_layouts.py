def test_get_array_layout():
    from vipy.layouts.layouts import get_array_layout

    layout = get_array_layout("eht")

    assert len(layout) == 8
    assert type(layout[0].name) == str
    assert type(layout[0].x) == float
    assert type(layout[0].y) == float
    assert type(layout[0].z) == float
    assert type(layout[0].diam) == float
    assert type(layout[0].el_low) == int
    assert type(layout[0].el_high) == int
    assert type(layout[0].sefd) == int
    assert type(layout[0].altitude) == int
