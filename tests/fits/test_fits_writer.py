import numpy as np
import pytest
import torch
from astropy.io import fits
from astropy.time import Time

from pyvisgen.fits.writer import (
    create_antenna_hdu,
    create_frequency_hdu,
    create_hdu_list,
    create_time_hdu,
    create_vis_hdu,
)


@pytest.fixture
def mock_obs(mocker):
    obs = mocker.MagicMock()
    obs.ra = torch.tensor(180.0)
    obs.dec = torch.tensor(45.0)
    obs.ref_frequency = torch.tensor(15.7e9)
    obs.bandwidths = torch.tensor([1.28e8, 1.28e8])
    obs.polarization = None
    obs.layout = "vlba"
    obs.start = Time("2025-12-01T00:00:00", format="isot", scale="utc")

    return obs


@pytest.fixture
def mock_obs_linear(mock_obs):
    mock_obs.polarization = "linear"

    return mock_obs


@pytest.fixture
def mock_vis(mocker):
    rng = np.random.default_rng()
    size = 8

    vis = mocker.MagicMock()
    vis.u = rng.standard_normal(size)
    vis.v = rng.standard_normal(size)
    vis.w = rng.standard_normal(size)

    vis.date = np.sort(rng.uniform(2460000.0, 2460001.0, size))
    vis.base_num = np.arange(1, size + 1)
    vis.num = np.arange(1, size + 1, dtype=np.int32)

    vals = rng.standard_normal((size, 2, 4)) + 1j * rng.standard_normal((size, 2, 4))

    vis.get_values.return_value = torch.from_numpy(vals)
    vis.weights = torch.ones(size, 1)

    return vis


class TestCreateVisHDU:
    def test_returns_groups_hdu(self, mock_vis, mock_obs) -> None:
        hdu = create_vis_hdu(mock_vis, mock_obs)

        assert isinstance(hdu, fits.GroupsHDU)

    def test_header_keys(self, mock_vis, mock_obs) -> None:
        hdu = create_vis_hdu(mock_vis, mock_obs)
        header = hdu.header

        assert header["EXTNAME"] == "AIPS UV"
        assert header["OBJECT"] == "sim-source-0"  # default name
        assert header["TELESCOP"] == header["INSTRUME"] == "vlba"
        assert header["BUNIT"] == "UNCALIB"
        assert header["EPOCH"] == 2000.0
        assert header["BSCALE"] == 1.0
        assert header["BZERO"] == 0.0
        assert header["CRVAL3"] == -1
        np.testing.assert_almost_equal(header["OBSRA"], 180.0)
        np.testing.assert_almost_equal(header["OBSDEC"], 45.0)

    def test_custom_source_name(self, mock_vis, mock_obs) -> None:
        hdu = create_vis_hdu(mock_vis, mock_obs, source_name="M87")

        assert hdu.header["OBJECT"] == "M87"

    def test_wcs_stokes_linear(self, mock_vis, mock_obs_linear) -> None:
        hdu = create_vis_hdu(mock_vis, mock_obs_linear)

        assert hdu.header["CRVAL3"] == -5

    def test_pscale_pzero(self, mock_vis, mock_obs) -> None:
        hdu = create_vis_hdu(mock_vis, mock_obs)

        for i in range(1, 8):
            assert hdu.header[f"PSCAL{i}"] == 1.0
            assert hdu.header[f"PZERO{i}"] == 0.0

    def test_data_shape(self, mock_vis, mock_obs) -> None:
        hdu = create_vis_hdu(mock_vis, mock_obs)

        assert hdu.data.shape[0] == mock_vis.u.shape[0]


class TestCreateTimeHDU:
    def test_returns_bintable(self, mock_vis) -> None:
        hdu = create_time_hdu(mock_vis)

        assert isinstance(hdu, fits.BinTableHDU)

    def test_column_names(self, mock_vis) -> None:
        hdu = create_time_hdu(mock_vis)

        expected = {
            "TIME",
            "TIME INTERVAL",
            "SOURCE ID",
            "SUBARRAY",
            "FREQ ID",
            "START VIS",
            "END VIS",
        }

        assert set(hdu.columns.names) == expected

    def test_vis_start_end(self, mock_vis) -> None:
        hdu = create_time_hdu(mock_vis)

        assert hdu.data["START VIS"][0] == mock_vis.num.min()
        assert hdu.data["END VIS"][0] == mock_vis.num.max()

    def test_time_interval_nonnegative(self, mock_vis) -> None:
        hdu = create_time_hdu(mock_vis)

        assert hdu.data["TIME INTERVAL"][0] >= 0


class TestCreateFrequencyHDU:
    def test_returns_bintable(self, mock_obs) -> None:
        hdu = create_frequency_hdu(mock_obs)

        assert isinstance(hdu, fits.BinTableHDU)

    def test_column_names(self, mock_obs) -> None:
        hdu = create_frequency_hdu(mock_obs)

        expected = {
            "FRQSEL",
            "IF FREQ",
            "CH WIDTH",
            "TOTAL BANDWIDTH",
            "SIDEBAND",
        }

        assert set(hdu.columns.names) == expected

    def test_total_bandwidth(self, mock_obs) -> None:
        hdu = create_frequency_hdu(mock_obs)

        expected = (mock_obs.bandwidths[0] * len(mock_obs.bandwidths)).cpu().numpy()

        np.testing.assert_almost_equal(hdu.data["TOTAL BANDWIDTH"][0], expected)


class TestCreateAntennaHDU:
    def test_returns_bintable(self, mock_obs) -> None:
        hdu = create_antenna_hdu(mock_obs)

        assert isinstance(hdu, fits.BinTableHDU)

    def test_column_names(self, mock_obs) -> None:
        hdu = create_antenna_hdu(mock_obs)

        expected = {
            "ANNAME",
            "STABXYZ",
            "ORBPARM",
            "NOSTA",
            "MNTSTA",
            "STAXOF",
            "POLTYA",
            "POLAA",
            "POLCALA",
            "POLTYB",
            "POLAB",
            "POLCALB",
            "DIAMETER",
        }

        assert set(hdu.columns.names) == expected

    def test_header_keys(self, mock_obs) -> None:
        hdu = create_antenna_hdu(mock_obs)
        header = hdu.header

        assert header["ARRNAM"] == "vlba"
        assert header["TIMSYS"] == "UTC"
        assert header["FRAME"] == "ITRF"
        assert header["NUMORB"] == 0
        assert header["NOPCAL"] == 2
        assert header["NO_IF"] == 1

    def test_poltype(self, mock_obs) -> None:
        hdu = create_antenna_hdu(mock_obs)

        poltya = hdu.data["POLTYA"]
        poltyb = hdu.data["POLTYB"]

        assert all([p == "X" for p in poltya])
        assert all([p == "Y" for p in poltyb])

    def test_orbparm(self, mock_obs) -> None:
        hdu = create_antenna_hdu(mock_obs)

        assert hdu.data["ORBPARM"].size == 0
        assert hdu.data["ORBPARM"].dtype is np.dtype("float64")


class TestCreateHDUList:
    @pytest.fixture
    def mock_hdus(self, mocker) -> tuple:
        mock_create_vis_hdu = mocker.patch(
            "pyvisgen.fits.writer.create_vis_hdu",
            return_value=fits.GroupsHDU(),
        )
        mock_create_time_hdu = mocker.patch(
            "pyvisgen.fits.writer.create_time_hdu",
            return_value=fits.BinTableHDU(),
        )
        mock_create_frequency_hdu = mocker.patch(
            "pyvisgen.fits.writer.create_frequency_hdu",
            return_value=fits.BinTableHDU(),
        )
        mock_create_antenna_hdu = mocker.patch(
            "pyvisgen.fits.writer.create_antenna_hdu",
            return_value=fits.BinTableHDU(),
        )

        return (
            mock_create_vis_hdu,
            mock_create_time_hdu,
            mock_create_frequency_hdu,
            mock_create_antenna_hdu,
        )

    def test_returns_hdulist(self, mock_hdus: tuple, mock_vis, mock_obs) -> None:
        mock_hdus  # noqa: B018
        hdu_list = create_hdu_list(mock_vis, mock_obs)

        assert isinstance(hdu_list, fits.HDUList)
        assert len(hdu_list) == 4

    def test_functions_called(self, mock_hdus: tuple, mock_vis, mock_obs) -> None:
        (
            mock_create_vis_hdu,
            mock_create_time_hdu,
            mock_create_frequency_hdu,
            mock_create_antenna_hdu,
        ) = mock_hdus

        create_hdu_list(mock_vis, mock_obs)

        mock_create_vis_hdu.assert_called_once_with(mock_vis, mock_obs)
        mock_create_time_hdu.assert_called_once_with(mock_vis)
        mock_create_frequency_hdu.assert_called_once_with(mock_obs)
        mock_create_antenna_hdu.assert_called_once_with(mock_obs)

    def test_hdu_order(self, mock_vis, mock_obs) -> None:
        hdu_list = create_hdu_list(mock_vis, mock_obs)

        assert isinstance(hdu_list[0], fits.GroupsHDU)

        for i, ext in enumerate(["AIPS FQ", "AIPS AN", "AIPS NX"], start=1):
            assert isinstance(hdu_list[i], fits.BinTableHDU)
            assert hdu_list[i].header["EXTNAME"] == ext
