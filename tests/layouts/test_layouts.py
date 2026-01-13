from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyvisgen.layouts import Stations, get_array_layout


class TestStations:
    @pytest.fixture
    def params(self) -> dict:
        rng = np.random.default_rng()

        params = dict(
            st_num=np.arange(10),
            x=rng.uniform(size=(10)),
            y=rng.uniform(size=(10)),
            z=rng.uniform(size=(10)),
            diam=np.full(shape=(10), fill_value=5),
            el_low=np.full(shape=(10), fill_value=15),
            el_high=np.full(shape=(10), fill_value=85),
            altitude=rng.uniform(1e4, 8e4, size=(10)),
            sefd=np.full(shape=(10), fill_value=100),
        )

        return params

    def test_stations_instantiation(self, params: dict) -> None:
        # should not raise an exception
        Stations(**params)

    def test_getitem(self, params: dict) -> None:
        stations = Stations(**params)

        for item, param in zip(stations[0], params):
            assert item == param[0]


class TestGetArrayLayout:
    def test_array_layout_str(self) -> None:
        stations = get_array_layout(array_layout="vlba")

        assert isinstance(stations, Stations)

    def test_array_layout_invalid_str(self) -> None:
        with pytest.raises(FileNotFoundError) as excinfo:
            get_array_layout(array_layout="this_array_does_not_exist")

        assert "No such file or directory" in str(excinfo.value)

    def test_array_layout_path(self) -> None:
        stations = get_array_layout(array_layout=Path("tests/data/test_layout.txt"))

        assert isinstance(stations, Stations)

    def test_array_layout_dataframe(self) -> None:
        df = pd.read_csv("tests/data/test_layout.txt", sep=r"\s+")
        stations = get_array_layout(array_layout=df)

        assert isinstance(stations, Stations)

    def test_array_layout_invalid(self) -> None:
        with pytest.raises(TypeError) as excinfo:
            get_array_layout(array_layout=None)

        assert "Expected array_layout to be of type" in str(excinfo.value)

    def test_writer(self) -> None:
        df = pd.read_csv("tests/data/test_layout.txt", sep=r"\s+")
        stations = get_array_layout(array_layout=df, writer=True)

        assert isinstance(stations, pd.DataFrame)
        pd.testing.assert_frame_equal(stations, df)
