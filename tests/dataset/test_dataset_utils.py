import numpy as np
import pytest

from pyvisgen.dataset.utils import calc_truth_fft, convert_amp_phase, convert_real_imag


class TestUtils:
    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng()

    @pytest.fixture
    def stacked_data(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(size=(1, 2, 32, 32))

    @pytest.fixture
    def cmplx_data(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(size=(1, 1, 32, 32)) + 1j * rng.uniform(size=(1, 1, 32, 32))

    def test_calc_truth_fft(self) -> None:
        arr = np.ones((1, 1, 32, 32))

        result = calc_truth_fft(arr)

        expected = np.zeros((1, 1, 32, 32)) + 1j * np.zeros((1, 1, 32, 32))
        expected[..., 16, 16] = 1024 + 1j * 0

        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("sky_sim", [True, False])
    def test_convert_amp_phase(
        self, sky_sim: bool, stacked_data: np.ndarray, cmplx_data: np.ndarray
    ) -> None:
        data = cmplx_data if sky_sim else stacked_data

        result = convert_amp_phase(data, sky_sim=sky_sim)

        data = data[:, 0] + 1j * data[:, 1] if not sky_sim else data[0]

        np.testing.assert_allclose(result[:, 0], np.abs(data))
        np.testing.assert_allclose(result[:, 1], np.angle(data))

    @pytest.mark.parametrize("sky_sim", [True, False])
    def test_convert_real_imag(
        self, sky_sim: bool, stacked_data: np.ndarray, cmplx_data: np.ndarray
    ) -> None:
        data = cmplx_data if sky_sim else stacked_data

        result = convert_real_imag(data, sky_sim=sky_sim)

        real = data.real[0] if sky_sim else data[:, 0]
        imag = data.imag[0] if sky_sim else data[:, 1]

        np.testing.assert_allclose(result[:, 0], real)
        np.testing.assert_allclose(result[:, 1], imag)
