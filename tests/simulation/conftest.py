import pytest
import torch
from astropy.time import Time

from pyvisgen.simulation.observation import Baselines, ValidBaselineSubset


@pytest.fixture(scope="module")
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def baselines_data() -> dict[str, torch.Tensor]:
    size = 10
    return {
        "st1": torch.arange(size),
        "st2": torch.arange(size),
        "u": torch.rand(size),
        "v": torch.rand(size),
        "w": torch.rand(size),
        "valid": torch.randint(low=0, high=1, size=(size,)),
        "time": torch.linspace(0, 1000, size),
        "q1": torch.rand(size),
        "q2": torch.rand(size),
    }


@pytest.fixture(scope="module")
def subset_data(device) -> dict[str, torch.Tensor]:
    size = 10

    dev = torch.device(device)

    time = Time(torch.linspace(0, 1000, size) / (60 * 60 * 24), format="mjd").jd
    date = (torch.from_numpy(time) / 2).to(device)

    return {
        "u_start": torch.rand(size, device=dev),
        "u_stop": torch.rand(size, device=dev),
        "u_valid": torch.rand(size, device=dev),
        "v_start": torch.rand(size, device=dev),
        "v_stop": torch.rand(size, device=dev),
        "v_valid": torch.rand(size, device=dev),
        "w_start": torch.rand(size, device=dev),
        "w_stop": torch.rand(size, device=dev),
        "w_valid": torch.rand(size, device=dev),
        "baseline_nums": torch.arange(size, device=dev),
        "date": date,
        "q1_start": torch.rand(size, device=dev),
        "q1_stop": torch.rand(size, device=dev),
        "q1_valid": torch.rand(size, device=dev),
        "q2_start": torch.rand(size, device=dev),
        "q2_stop": torch.rand(size, device=dev),
        "q2_valid": torch.rand(size, device=dev),
    }


@pytest.fixture(scope="module")
def baselines(baselines_data: dict[str, torch.Tensor]) -> Baselines:
    return Baselines(**baselines_data)


@pytest.fixture(scope="module")
def subset(subset_data: dict[str, torch.Tensor]) -> ValidBaselineSubset:
    return ValidBaselineSubset(**subset_data)
