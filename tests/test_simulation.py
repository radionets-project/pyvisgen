import numpy as np
from pyvisgen.utils.config import read_data_set_conf
from pathlib import Path

np.random.seed(1)
config = "tests/test_conf.toml"
conf = read_data_set_conf(config)
out_path = Path(conf["out_path"])
out_path.mkdir(parents=True, exist_ok=True)


def test_get_data():
    from radiosim.data import radiosim_data

    data = radiosim_data(conf["in_path"])
    assert len(data) > 0


def test_create_sampling_rc():
    from pyvisgen.simulation.data_set import test_opts, create_sampling_rc

    samp_ops = create_sampling_rc(conf)
    assert len(samp_ops) == 14

    test_opts(samp_ops)
    # assert active_telescopes > 0
