from pathlib import Path

import torch

from pyvisgen.utils.config import read_data_set_conf

torch.manual_seed(1)
config = "tests/test_conf.toml"
conf = read_data_set_conf(config)
out_path = Path(conf["out_path_fits"])
out_path.mkdir(parents=True, exist_ok=True)


def test_get_data():
    from pyvisgen.utils.data import load_bundles

    data = load_bundles(conf["in_path"])
    assert len(data) > 0


def test_create_sampling_rc():
    from pyvisgen.simulation.data_set import create_sampling_rc, test_opts

    samp_ops = create_sampling_rc(conf)
    assert len(samp_ops) == 17

    test_opts(samp_ops)


def test_vis_loop():
    import torch

    from pyvisgen.simulation.data_set import create_observation
    from pyvisgen.simulation.visibility import vis_loop
    from pyvisgen.utils.data import load_bundles, open_bundles

    bundles = load_bundles(conf["in_path"])
    obs = create_observation(conf)
    # num_active_telescopes = test_opts(samp_ops)
    data = open_bundles(bundles[0])
    SI = torch.tensor(data[0])[None]
    vis_data = vis_loop(obs, SI, noisy=conf["noisy"], mode=conf["mode"])

    assert (vis_data[0].SI[0]).dtype == torch.complex128
    assert (vis_data[0].SQ[0]).dtype == torch.complex128
    assert (vis_data[0].SU[0]).dtype == torch.complex128
    assert (vis_data[0].SV[0]).dtype == torch.complex128
    assert (vis_data[0].num).dtype == torch.float32
    assert (vis_data[0].base_num).dtype == torch.float64
    assert torch.is_tensor(vis_data[0].u)
    assert torch.is_tensor(vis_data[0].v)
    assert torch.is_tensor(vis_data[0].w)
    assert (vis_data[0].date).dtype == torch.float64

    # test num vis for time step 0
    # num_vis_theory = num_active_telescopes * (num_active_telescopes - 1) / 2
    # num_vis_calc = vis_data.base_num[vis_data.date == vis_data.date[0]].shape[0]
    # dunno what's going on here
    # assert num_vis_theory == num_vis_calc
