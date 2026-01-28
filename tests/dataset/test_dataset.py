from datetime import datetime
from logging import INFO, WARNING
from unittest.mock import call, patch

import numpy as np
import pytest
import torch

from pyvisgen.dataset import SimulateDataSet
from pyvisgen.io import Config
from pyvisgen.layouts import Stations


class TestGetGridder:
    @pytest.fixture
    def sd(self, simulate_dataset: SimulateDataSet) -> SimulateDataSet:
        simulate_dataset.conf = Config.from_toml("tests/test_conf.toml")
        return simulate_dataset

    def test_get_from_plugin_manager(self, caplog, sd: SimulateDataSet) -> None:
        sd.conf.gridding.gridder = "pyvisgrid.gridder"

        with caplog.at_level(WARNING):
            sd._get_gridder()

        assert "Falling back to default gridder" not in caplog.text

    def test_fallback(self, caplog, sd: SimulateDataSet) -> None:
        sd.conf.gridding.gridder = "this_gridder_does_not_exist"

        with caplog.at_level(WARNING):
            sd._get_gridder()

        assert "Falling back to default gridder" in caplog.text


class TestCreateObservation:
    @pytest.fixture
    def sd(self, simulate_dataset: SimulateDataSet) -> SimulateDataSet:
        simulate_dataset.conf = Config.from_toml("tests/test_conf.toml")

        # simpler sample options for test
        simulate_dataset.samp_opts = dict(
            src_ra=torch.tensor([180.0]),
            src_dec=torch.tensor([45.0]),
            start_time=np.array([datetime(2025, 1, 1, 12)]).astype(datetime),
            scan_duration=torch.tensor([60]),
            num_scans=torch.tensor([2]),
            delta=torch.tensor([90]),
            amp_ratio=torch.tensor([0.5]),
            order=torch.tensor([0.5, 0.5]),
            scale=torch.tensor([0.5, 1.0]),
            threshold=None,
        )

        simulate_dataset.samp_opts_const = dict(
            array_layout="vlba",
            image_size=128,
            fov=0.14,
            integration_time=10.0,
            scan_separation=4500,
            ref_frequency=15e9,
            frequency_offsets=[0.0],
            bandwidths=[1.28e8],
            corrupted=False,
            polarization=None,
        )

        return simulate_dataset

    def test_observation(self, sd: SimulateDataSet) -> None:
        with patch(target="pyvisgen.dataset.dataset.Observation") as mock_obs:
            sd.create_observation(0)

            mock_obs.assert_called_once()
            call_kwargs = mock_obs.call_args.kwargs

            # Check that correct kwargs were used in the call
            # (small sample)
            assert call_kwargs["src_ra"] == 180.0
            assert call_kwargs["src_dec"] == 45.0
            assert call_kwargs["scan_duration"] == 60
            assert call_kwargs["num_scans"] == 2

    def test_observation_dense(self, sd: SimulateDataSet) -> None:
        sd.conf.sampling.mode = "dense"

        with patch(target="pyvisgen.dataset.dataset.Observation") as mock_obs:
            sd.create_observation(0)

            mock_obs.assert_called_once()
            call_kwargs = mock_obs.call_args.kwargs

            assert call_kwargs["dense"] is True


class TestCreateSamplingRC:
    def test_samp_opts_const_keys(self, sd_sampling: SimulateDataSet) -> None:
        sd_sampling.create_sampling_rc(1)

        expected_keys = {
            "array_layout",
            "image_size",
            "fov",
            "integration_time",
            "scan_separation",
            "ref_frequency",
            "frequency_offsets",
            "bandwidths",
            "corrupted",
            "device",
            "sensitivity_cut",
            "polarization",
        }

        assert set(sd_sampling.samp_opts_const) == expected_keys

    def test_instances(self, sd_sampling: SimulateDataSet) -> None:
        sd_sampling.create_sampling_rc(1)

        assert isinstance(sd_sampling.freq_bands, np.ndarray)
        assert isinstance(sd_sampling.array, Stations)

    def test_shape(self, sd_sampling: SimulateDataSet) -> None:
        sd_sampling.create_sampling_rc(size=10)

        assert sd_sampling.samp_opts["src_ra"].size()[0] == 10

    def test_calls(self, sd_sampling: SimulateDataSet) -> None:
        size = 10

        with (
            patch.object(
                sd_sampling,
                "draw_sampling_opts",
                return_value={"src_ra": torch.zeros(size)},
            ) as mock_draw,
            patch.object(sd_sampling, "test_rand_opts") as mock_test,
        ):
            sd_sampling.create_sampling_rc(size=size)

            mock_draw.assert_called_once_with(size)

            calls = [call(i) for i in range(size)]
            mock_test.assert_has_calls(calls, any_order=True)

    def test_no_seed(self, sd_sampling: SimulateDataSet) -> None:
        size = 1

        with (
            patch(
                "pyvisgen.dataset.dataset.np.random.default_rng",
                return_value=np.random.default_rng(),
            ) as rng,
            # mock sampling and tests to increase speed
            patch.object(
                sd_sampling,
                "draw_sampling_opts",
                return_value={"src_ra": torch.zeros(size)},
            ),
            patch.object(sd_sampling, "test_rand_opts"),
        ):
            sd_sampling.conf.sampling.seed = None
            sd_sampling.create_sampling_rc(size)

            rng.assert_called_once_with()

    def test_dense(self, sd_sampling: SimulateDataSet) -> None:
        ref_freq = np.array(sd_sampling.conf.sampling.ref_frequency)

        with (
            # mock sampling and tests to increase speed
            patch.object(
                sd_sampling,
                "draw_sampling_opts",
                return_value={"src_ra": torch.zeros(1)},
            ),
            patch.object(sd_sampling, "test_rand_opts"),
        ):
            sd_sampling.conf.sampling.mode = "dense"
            sd_sampling.create_sampling_rc(1)

            np.testing.assert_array_equal(sd_sampling.freq_bands, ref_freq)


class TestDrawSamplingOpts:
    def test_samp_opts_keys(self, sd_sampling: SimulateDataSet) -> None:
        samp_opts = sd_sampling.draw_sampling_opts(1)

        expected_keys = {
            "src_ra",
            "src_dec",
            "start_time",
            "scan_duration",
            "num_scans",
            "delta",
            "amp_ratio",
            "order",
            "scale",
            "threshold",
        }

        assert set(samp_opts.keys()) == expected_keys
        assert samp_opts["scan_duration"].dtype == torch.int64
        assert samp_opts["num_scans"].dtype == torch.int64

        # make sure that polarization is not enabled and polarization
        # related values such as delta are NaN
        assert torch.isnan(samp_opts["delta"]).all()

    def test_shapes(self, sd_sampling: SimulateDataSet):
        size = 10
        samp_opts = sd_sampling.draw_sampling_opts(size)

        opts = [
            "src_ra",
            "src_dec",
            "start_time",
            "scan_duration",
            "num_scans",
        ]
        for opt in opts:
            assert samp_opts[opt].shape == (size,)

    @pytest.mark.parametrize("pol_mode", ["linear", "circular"])
    def test_polarization(self, pol_mode: str, sd_sampling: SimulateDataSet) -> None:
        sd_sampling.conf.polarization.mode = pol_mode

        size = 10
        samp_opts = sd_sampling.draw_sampling_opts(size)

        assert not torch.isnan(samp_opts["delta"]).all()
        assert not torch.isnan(samp_opts["amp_ratio"]).all()
        assert not torch.isnan(samp_opts["order"]).all()
        assert not torch.isnan(samp_opts["scale"]).all()
        assert samp_opts["delta"].shape == (size,)
        assert samp_opts["amp_ratio"].shape == (size,)
        assert samp_opts["order"].shape == (size, 2)
        assert samp_opts["scale"].shape == (size, 2)

    def test_polarization_kwargs_none(self, sd_sampling: SimulateDataSet) -> None:
        sd_sampling.conf.polarization.mode = "linear"
        sd_sampling.conf.polarization.delta = None
        sd_sampling.conf.polarization.amp_ratio = None
        sd_sampling.conf.polarization.field_order = []
        sd_sampling.conf.polarization.field_scale = []

        size = 10
        samp_opts = sd_sampling.draw_sampling_opts(size)

        assert not torch.isnan(samp_opts["delta"]).all()
        assert not torch.isnan(samp_opts["amp_ratio"]).all()
        assert not torch.isnan(samp_opts["order"]).all()
        assert not torch.isnan(samp_opts["scale"]).all()
        assert samp_opts["delta"].shape == (size,)
        assert samp_opts["amp_ratio"].shape == (size,)
        assert samp_opts["order"].shape == (size, 2)
        assert samp_opts["scale"].shape == (size, 2)

    def test_reproducible(self, sd_sampling: SimulateDataSet) -> None:
        sd_sampling.rng = np.random.default_rng(42)
        opts1 = sd_sampling.draw_sampling_opts(10)

        sd_sampling.rng = np.random.default_rng(42)
        opts2 = sd_sampling.draw_sampling_opts(10)

        for opt1, opt2 in zip(opts1.values(), opts2.values()):
            np.testing.assert_array_equal(opt1, opt2)


class TestGetImages:
    @pytest.fixture
    def sd(self, simulate_dataset: SimulateDataSet) -> SimulateDataSet:
        simulate_dataset.data_paths = ["tests/data/test_inputs.h5"]
        simulate_dataset.key = "y"

        return simulate_dataset

    def test_shape_unsqueeze(self, sd: SimulateDataSet) -> None:
        image = sd.get_images(0)

        assert image.shape == (10, 1, 128, 128)

    def test_output_is_tensor(self, sd: SimulateDataSet) -> None:
        image = sd.get_images(0)

        assert isinstance(image, torch.Tensor)


class TestGeocentricToSpherical:
    @pytest.mark.parametrize(
        ["x", "y", "z", "expected_lon", "expected_lat"],
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 90.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 90.0],
            [0.0, 0.0, -1.0, 0.0, -90.0],
        ],
    )
    def test_directions(
        self,
        simulate_dataset: SimulateDataSet,
        x: float | torch.Tensor,
        y: float | torch.Tensor,
        z: float | torch.Tensor,
        expected_lon: float,
        expected_lat: float,
    ) -> None:
        x = torch.as_tensor([x])
        y = torch.as_tensor([y])
        z = torch.as_tensor([z])

        lat, lon = simulate_dataset._geocentric_to_spherical(x, y, z)

        assert lon.item() == expected_lon
        assert lat.item() == expected_lat

    def test_batches(self, simulate_dataset: SimulateDataSet) -> None:
        x = torch.as_tensor([1.0, 0.0, 0.0])
        y = torch.as_tensor([0.0, 1.0, 0.0])
        z = torch.as_tensor([0.0, 0.0, 1.0])

        lat, lon = simulate_dataset._geocentric_to_spherical(x, y, z)

        assert lon.shape == (3,)
        assert lat.shape == (3,)


class TestComputeAltitude:
    @pytest.fixture
    def sd(self, simulate_dataset: SimulateDataSet) -> SimulateDataSet:
        simulate_dataset.array_lat = torch.tensor([45.0])

        return simulate_dataset

    @pytest.mark.parametrize(
        ["ra", "dec", "lst", "expected_alt"],
        [
            [0.0, 45.0, 0.0, 90.0],  # Zenith for array at lat = 45
            [0.0, -45.0, 0.0, 0.0],  # Horizon for array at lat = 45
        ],
    )
    def test_zenith_horizon(
        self,
        ra: int | float | torch.Tensor,
        dec: int | float | torch.Tensor,
        lst: int | float | torch.Tensor,
        expected_alt: int | float | torch.Tensor,
        sd: SimulateDataSet,
    ) -> None:
        ra = torch.tensor([ra])
        dec = torch.tensor([dec])
        lst = torch.tensor([lst])

        alt = sd._compute_altitude(ra, dec, lst)

        assert alt.item() == pytest.approx(expected_alt, abs=1e-8)

    def test_lst(self, sd: SimulateDataSet) -> None:
        ra = torch.tensor([0])
        dec = torch.tensor([45])
        lst = torch.tensor([0, 90, 180])

        alt = sd._compute_altitude(ra, dec, lst)

        assert alt.shape == (3,)

    @pytest.mark.parametrize(["lat", "dec"], [[90, 90], [-90, -90]])
    def test_poles(
        self,
        lat: int | float,
        dec: int | float | torch.Tensor,
        simulate_dataset: SimulateDataSet,
    ) -> None:
        simulate_dataset.array_lat = torch.tensor([lat])

        ra = torch.tensor([0])
        dec = torch.tensor([dec])
        lst = torch.tensor([0])

        alt = simulate_dataset._compute_altitude(ra, dec, lst)

        assert not torch.isnan(alt).any()
        assert alt.item() == pytest.approx(90.0, abs=1e-8)


@patch.object(SimulateDataSet, "_run")
@patch.object(SimulateDataSet, "get_images", return_value=[1])
class TestFromConfig:
    def test_conf_path(self, mock_get_images, mock_run) -> None:
        sd = SimulateDataSet.from_config("tests/test_conf.toml")

        assert isinstance(sd.conf, Config)

    def test_conf_config(self, mock_get_images, mock_run) -> None:
        cfg = Config.from_toml("tests/test_conf.toml")
        sd = SimulateDataSet.from_config(cfg)

        assert isinstance(sd.conf, Config)

    def test_conf_dict(self, mock_get_images, mock_run) -> None:
        cfg_dict: dict = Config.from_toml("tests/test_conf.toml").model_dump()
        sd = SimulateDataSet.from_config(cfg_dict)

        assert isinstance(sd.conf, Config)

    def test_conf_invalid(self, mock_get_images, mock_run) -> None:
        cfg = None

        with pytest.raises(ValueError) as excinfo:
            SimulateDataSet.from_config(cfg)

        assert (
            "Expected config to be one of str, Path, dict, or pyvisgen.io.Config!"
            in str(excinfo.value)
        )

    def test_multiprocess_all(self, mock_get_images, mock_run) -> None:
        sd = SimulateDataSet.from_config("tests/test_conf.toml", multiprocess="all")

        assert sd.multiprocess == -1

    def test_no_images_found(self, mock_get_images, mock_run) -> None:
        mock_get_images.return_value = []

        with pytest.raises(ValueError) as excinfo:
            SimulateDataSet.from_config("tests/test_conf.toml")

        assert "No images found in bundles! Please check your input path!" in str(
            excinfo.value
        )


class TestRun:
    @pytest.fixture
    def sd_run(self, mocker, sd_sampling: SimulateDataSet) -> SimulateDataSet:
        sd_sampling.data_paths = [None, None]  # 2 bundles
        sd_sampling.writer = None  # dummy to add writer attribute for mocks
        sd_sampling.gridder = None  # dummy to add gridder attribute for mocks

        mocker.patch.object(
            sd_sampling, "get_images", return_value=torch.zeros((10, 1, 32, 32))
        )
        mocker.patch.object(sd_sampling, "create_observation")
        mocker.patch("pyvisgen.dataset.dataset.vis_loop")
        mocker.patch.object(sd_sampling, "writer")
        mocker.patch.object(sd_sampling, "gridder")

        return sd_sampling

    @pytest.mark.parametrize("amp_phase", [True, False])
    def test_grid(
        self, amp_phase: bool, mocker, caplog, sd_run: SimulateDataSet
    ) -> None:
        sd_run.grid = True
        sd_run.stokes_comp = "I"
        sd_run.conf.bundle.amp_phase = amp_phase

        mock_gridder = mocker.patch.object(sd_run.gridder, "from_pyvisgen")
        mock_writer = mocker.patch.object(sd_run.writer, "write")
        mock_convert_amp_phase = mocker.patch(
            "pyvisgen.dataset.dataset.convert_amp_phase",
            return_value=torch.zeros((10, 2, 32, 32)),
        )
        mock_convert_real_imag = mocker.patch(
            "pyvisgen.dataset.dataset.convert_real_imag",
            return_value=torch.zeros((10, 2, 32, 32)),
        )

        with caplog.at_level(INFO):
            sd_run._run()

        assert mock_gridder.called
        assert mock_writer.called

        if amp_phase:
            assert mock_convert_amp_phase.called
            assert not mock_convert_real_imag.called
        else:
            assert mock_convert_real_imag.called
            assert not mock_convert_amp_phase.called

        assert "Successfully simulated and saved" in caplog.text

    @pytest.mark.parametrize("amp_phase", [True, False])
    def test_grid_raise_wrong_shape(
        self, amp_phase: bool, mocker, sd_run: SimulateDataSet
    ) -> None:
        sd_run.grid = True
        sd_run.stokes_comp = "I"
        sd_run.conf.bundle.amp_phase = amp_phase

        mocker.patch.object(sd_run.gridder, "from_pyvisgen")
        mocker.patch.object(sd_run.writer, "write")
        mocker.patch(
            "pyvisgen.dataset.dataset.convert_amp_phase",
            return_value=torch.zeros((10, 1, 32, 32)),
        )
        mocker.patch(
            "pyvisgen.dataset.dataset.convert_real_imag",
            return_value=torch.zeros((10, 1, 32, 32)),
        )

        with pytest.raises(ValueError) as excinfo:
            sd_run._run()

        assert "Expected 'sim_data' axis at index 1 to be 2!" in str(excinfo.value)

    def test_no_grid(self, mocker, caplog, sd_run: SimulateDataSet) -> None:
        sd_run.grid = False
        sd_run.stokes_comp = "I"

        mock_gridder = mocker.patch.object(sd_run.gridder, "from_pyvisgen")
        mock_writer = mocker.patch.object(sd_run.writer, "write")

        with caplog.at_level(INFO):
            sd_run._run()

        assert not mock_gridder.called
        assert mock_writer.called

        assert "Successfully simulated and saved" in caplog.text
