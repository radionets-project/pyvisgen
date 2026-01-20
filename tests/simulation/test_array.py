import numpy as np
import pytest
import torch

from pyvisgen.simulation import Array


class TestArray:
    @pytest.fixture
    def array_layout(self, mocker):
        mock_array_layout = mocker.MagicMock()
        mock_array_layout.st_num = torch.arange(10)
        mock_array_layout.x = torch.rand(10)
        mock_array_layout.y = torch.rand(10)
        mock_array_layout.z = torch.rand(10)
        mock_array_layout.diam = torch.full(size=(10,), fill_value=5)
        mock_array_layout.el_low = torch.full(size=(10,), fill_value=15)
        mock_array_layout.el_high = torch.full(size=(10,), fill_value=85)
        mock_array_layout.altitude = torch.rand(10)
        mock_array_layout.sefd = np.full(shape=(10,), fill_value=100)

        return mock_array_layout

    def test_instantiation(self, array_layout) -> None:
        array = Array(array_layout)

        assert array.array_layout == array_layout

    def test_relative_pos(self, array_layout) -> None:
        array = Array(array_layout)
        delta_x, delta_y, delta_z = array.relative_pos

        assert delta_x.shape == delta_y.shape == delta_z.shape == (45, 1)

        combs = torch.combinations(array_layout.x)
        expected = (combs[:, 0] - combs[:, 1]).reshape(-1, 1)

        np.testing.assert_array_equal(delta_x, expected)

    def test_antenna_pairs(self, array_layout) -> None:
        array = Array(array_layout)
        st_num_pairs, els_low_pairs, els_high_pairs = array.antenna_pairs

        assert (
            st_num_pairs.shape == els_low_pairs.shape == els_high_pairs.shape == (45, 2)
        )

        expected = torch.combinations(array_layout.st_num)
        np.testing.assert_array_equal(st_num_pairs, expected)
