from types import SimpleNamespace

import numpy as np
import pytest
from astropy.coordinates import EarthLocation


@pytest.fixture
def params() -> dict:
    rng = np.random.default_rng(42)

    params = dict(
        st_name=np.asarray(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]),
        st_num=np.arange(10),
        x=rng.uniform(size=(10)) * 1e5,
        y=rng.uniform(size=(10)) * 1e5,
        z=rng.uniform(size=(10)) * 1e5,
        diam=np.full(shape=(10), fill_value=5),
        el_low=np.full(shape=(10), fill_value=15),
        el_high=np.full(shape=(10), fill_value=85),
        altitude=rng.uniform(1e2, 8e3, size=(10)),
        sefd=np.full(shape=(10), fill_value=100),
    )

    return params


@pytest.fixture(scope="function")
def mock_stations(params) -> SimpleNamespace:
    return SimpleNamespace(**params)


@pytest.fixture
def mock_locations(mock_stations):
    return EarthLocation(mock_stations.x, mock_stations.y, mock_stations.z, unit="m")
