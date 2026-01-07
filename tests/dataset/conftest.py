import numpy as np
import pytest

from pyvisgen.dataset.dataset import DATEFMT, SimulateDataSet
from pyvisgen.io import Config


@pytest.fixture
def simulate_dataset() -> SimulateDataSet:
    return SimulateDataSet()


@pytest.fixture
def sd_sampling(simulate_dataset: SimulateDataSet) -> SimulateDataSet:
    simulate_dataset.conf = Config.from_toml("tests/test_conf.toml")
    simulate_dataset.rng = np.random.default_rng()
    simulate_dataset.date_fmt = DATEFMT
    simulate_dataset.device = "cpu"
    simulate_dataset.multiprocess = -1
    simulate_dataset.overall_task_id = 1

    return simulate_dataset
