import numpy as np
from pyvisgen.utils.config import read_data_set_conf
from pathlib import Path

np.random.seed(1)
config = "tests/test_conf.toml"
conf = read_data_set_conf(config)
out_path = Path(conf["out_path_fits"])
out_path.mkdir(parents=True, exist_ok=True)


def test_get_data():
    from pyvisgen.utils.data import load_bundles

    data = load_bundles(conf["in_path"])
    assert len(data) > 0


def test_create_sampling_rc():
    from pyvisgen.simulation.data_set import test_opts, create_sampling_rc

    samp_ops = create_sampling_rc(conf)
    assert len(samp_ops) == 15

    test_opts(samp_ops)


def test_vis_loop():
    import torch
    from pyvisgen.utils.data import load_bundles, open_bundles
    from pyvisgen.simulation.data_set import create_sampling_rc, test_opts
    from pyvisgen.simulation.visibility import vis_loop
    from astropy import units as un

    bundles = load_bundles(conf["in_path"])
    samp_ops = create_sampling_rc(conf)
    num_active_telescopes = test_opts(samp_ops)
    data = open_bundles(bundles[0])
    SI = torch.tensor(data[0], dtype=torch.cdouble)
    vis_data = vis_loop(samp_ops, SI)

    assert type(vis_data[0].SI[0]) == np.complex128
    assert type(vis_data[0].SQ[0]) == np.complex128
    assert type(vis_data[0].SU[0]) == np.complex128
    assert type(vis_data[0].SV[0]) == np.complex128
    assert type(vis_data[0].num) == np.float64
    assert type(vis_data[0].scan) == np.float64
    assert type(vis_data[0].base_num) == np.float64
    assert type(vis_data[0].u) == un.Quantity
    assert type(vis_data[0].v) == un.Quantity
    assert type(vis_data[0].w) == un.Quantity
    assert type(vis_data[0].date) == np.float64
    assert type(vis_data[0]._date) == np.float64

    # test num vis for time step 0
    num_vis_theory = num_active_telescopes * (num_active_telescopes - 1) / 2
    num_vis_calsc = vis_data.base_num[
        vis_data.date == np.unique(vis_data.date)[0]
    ].shape[0]

    assert num_vis_theory == num_vis_calsc
