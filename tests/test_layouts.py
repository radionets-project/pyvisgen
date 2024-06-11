import torch


def test_get_array_layout():
    from pyvisgen.layouts.layouts import get_array_layout

    layout = get_array_layout("eht")

    assert len(layout.st_num) == 8
    assert torch.is_tensor(layout[0].x)
    assert torch.is_tensor(layout[0].y)
    assert torch.is_tensor(layout[0].z)
    assert torch.is_tensor(layout[0].diam)
    assert torch.is_tensor(layout[0].el_low)
    assert torch.is_tensor(layout[0].el_high)
    assert torch.is_tensor(layout[0].sefd)
    assert torch.is_tensor(layout[0].altitude)

    layout = get_array_layout("vlba")

    assert len(layout.st_num) == 10

    layout = get_array_layout("vla")

    assert len(layout.st_num) == 27
    assert layout[:3].st_num.shape == torch.Size([3])
