import datetime

import pytest
import torch
from astropy.time import Time

from pyvisgen.simulation.observation import (
    Baselines,
    Observation,
    Scan,
    ValidBaselineSubset,
)
from pyvisgen.simulation.visibility import Polarization, Visibilities


@pytest.fixture
def sky_dist() -> torch.Tensor:
    return torch.rand(size=(1, 32, 32))


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


@pytest.fixture(scope="module")
def scan() -> Scan:
    start = Time("2026-01-21T00:00:00", format="isot", scale="utc")
    stop = Time("2026-01-21T03:00:00", format="isot", scale="utc")

    return Scan(start=start, stop=stop, separation=4500.0, integration_time=60.0)


@pytest.fixture(scope="module")
def obs_params() -> dict:
    return {
        "src_ra": 180.0,
        "src_dec": 45.0,
        "start_time": datetime.datetime(2026, 1, 21, 0, 0, 0),
        "scan_duration": 400,
        "num_scans": 6,
        "scan_separation": 120,
        "integration_time": 60,
        "ref_frequency": 15.7e9,
        "frequency_offsets": [0.0],
        "bandwidths": [1e8],
        "fov": 0.1,
        "image_size": 64,
        "array_layout": "vlba",
        "corrupted": False,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@pytest.fixture(scope="module")
def obs(obs_params: dict) -> Observation:
    return Observation(**obs_params)


@pytest.fixture(scope="module")
def visibilities_data(device: str) -> dict:
    size = 10
    img_size = 32
    dev = torch.device(device)

    time = Time(torch.linspace(0, 1000, size) / (60 * 60 * 24), format="mjd").jd
    date = (torch.from_numpy(time) / 2).to(dev)

    return {
        "V_11": torch.rand(size=(size, 1), device=dev),
        "V_22": torch.rand(size=(size, 1), device=dev),
        "V_12": torch.rand(size=(size, 1), device=dev),
        "V_21": torch.rand(size=(size, 1), device=dev),
        "num": torch.rand(size, device=dev),
        "base_num": torch.rand(size, device=dev),
        "u": torch.rand(size, device=dev),
        "v": torch.rand(size, device=dev),
        "w": torch.rand(size, device=dev),
        "date": date,
        "linear_dop": torch.rand((img_size, img_size), device=dev),
        "circular_dop": torch.rand((img_size, img_size), device=dev),
    }


@pytest.fixture(scope="module")
def visibilities(visibilities_data: dict) -> Visibilities:
    return Visibilities(**visibilities_data)


@pytest.fixture(scope="module")
def field_kwargs() -> dict:
    return {
        "order": 1,
        "random_state": None,
        "scale": None,
        "threshold": None,
    }


@pytest.fixture(scope="function")
def polarization_data(device: str, field_kwargs: dict) -> dict:
    img_size = 32
    dev = torch.device(device)

    return {
        "SI": torch.rand((1, img_size, img_size), device=dev),
        "sensitivity_cut": torch.rand(1, device=dev),
        "amp_ratio": torch.rand(1, device=dev),
        "delta": torch.randint(low=0, high=90, size=(1,), device=dev),
        "polarization": None,
        "field_kwargs": field_kwargs,
        "random_state": None,
        "device": dev,
    }


@pytest.fixture(scope="function")
def polarization(polarization_data: dict) -> Polarization:
    return Polarization(**polarization_data)
