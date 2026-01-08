from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from pyvisgen.io.config import (
    BundleConfig,
    CodeCarbonEmissionTrackerConfig,
    Config,
    DataWriterConfig,
    FFTConfig,
    GriddingConfig,
    PolarizationConfig,
    SamplingConfig,
)


class TestBundleConfig:
    def test_keys(self) -> None:
        cfg = BundleConfig()
        expected_keys = {
            "dataset_type",
            "in_path",
            "out_path",
            "overlap",
            "grid_size",
            "grid_fov",
            "amp_phase",
        }

        assert set(cfg.model_dump()) == expected_keys

    @pytest.mark.parametrize(
        "key,value",
        [
            (key, value)
            for key, values in [
                ("grid_size", [0, -1, -64]),
                ("grid_fov", [0, -0.1, -10]),
            ]
            for value in values
        ],
    )
    def test_invalid_default_field(self, key: str, value: list) -> None:
        with pytest.raises(ValidationError):
            BundleConfig(**{key: value})

    @pytest.mark.parametrize("key", ["in_path", "out_path"])
    def test_expand_path(self, key: str):
        path = "~/test"

        cfg = BundleConfig(**{key: path}).model_dump()

        assert isinstance(cfg[key], Path)
        assert cfg[key] == Path(path).expanduser().resolve()

    @pytest.mark.parametrize("path", ["none", ""])
    def test_expand_path_invalid(self, path):
        with pytest.raises(ValueError) as excinfo:
            BundleConfig(in_path=path)

        assert "cannot be empty!" in str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            BundleConfig(in_path=path)

        assert "cannot be empty!" in str(excinfo.value)


class TestCodeCarbonEmissionTrackerConfig:
    def test_keys(self) -> None:
        cfg = CodeCarbonEmissionTrackerConfig()
        expected_keys = {
            "log_level",
            "country_iso_code",
            "output_dir",
        }

        assert set(cfg.model_dump()) == expected_keys


class TestDataWriterConfig:
    @pytest.fixture
    def dw(self):
        from pyvisgen.io import datawriters

        return datawriters

    def test_keys(self) -> None:
        cfg = DataWriterConfig()
        expected_keys = {
            "writer",
            "overlap",
            "shard_pattern",
            "compress",
        }

        assert set(cfg.model_dump()) == expected_keys

    @pytest.mark.parametrize("overlap", [0, -1, -10])
    def test_invalid_overlap(self, overlap: int) -> None:
        with pytest.raises(ValidationError):
            DataWriterConfig(overlap=overlap)

    def test_setup_writer_callable(self, dw):
        cfg = DataWriterConfig(writer=dw.H5Writer)

        assert isinstance(cfg.writer, Callable)
        assert issubclass(cfg.writer, dw.DataWriter)
        assert cfg.writer == dw.H5Writer

    def test_setup_writer_str(self, dw):
        cfg = DataWriterConfig(writer="H5Writer")

        assert isinstance(cfg.writer, Callable)
        assert issubclass(cfg.writer, dw.DataWriter)
        assert cfg.writer == dw.H5Writer

    @pytest.mark.parametrize(
        "shorthand,writer_name",
        [
            ("h5", "H5Writer"),
            ("hdf5", "H5Writer"),
            ("HDF5", "H5Writer"),
            ("wds", "WDSShardWriter"),
            ("WDS", "WDSShardWriter"),
            ("webdataset", "WDSShardWriter"),
            ("pt", "PTWriter"),
            ("PT", "PTWriter"),
        ],
    )
    def test_setup_writer_shorthands(self, shorthand, writer_name, dw):
        cfg = DataWriterConfig(writer=shorthand)

        expected_writer = getattr(dw, writer_name)

        assert isinstance(cfg.writer, Callable)
        assert issubclass(cfg.writer, dw.DataWriter)
        assert cfg.writer == expected_writer


class TestFFTConfig:
    def test_keys(self) -> None:
        cfg = FFTConfig()
        expected_keys = {"ft"}

        assert set(cfg.model_dump()) == expected_keys

    @pytest.mark.parametrize("ft", ["default", "finufft", "reversed"])
    def test_valid_mode(self, ft: str) -> None:
        cfg = FFTConfig(ft=ft)

        assert cfg.ft == ft

    def test_invalid_mode(self) -> None:
        with pytest.raises(ValidationError):
            FFTConfig(ft="invalid")


class TestGriddingConfig:
    def test_keys(self) -> None:
        cfg = GriddingConfig()
        expected_keys = {"gridder"}

        assert set(cfg.model_dump()) == expected_keys


class TestPolarizationConfig:
    def test_keys(self) -> None:
        cfg = PolarizationConfig()
        expected_keys = {
            "mode",
            "delta",
            "amp_ratio",
            "field_order",
            "field_scale",
            "field_threshold",
        }

        assert set(cfg.model_dump()) == expected_keys

    @pytest.mark.parametrize(
        "amp_ratio",
        [-1e-5, -1, 1.1, 10],
    )
    def test_invalid_amp_ratio(self, amp_ratio) -> None:
        with pytest.raises(ValidationError):
            PolarizationConfig(amp_ratio=amp_ratio)

    @pytest.mark.parametrize("key", ["mode", "delta", "amp_ratio", "field_threshold"])
    def test_parse_mode_thres(self, key: str) -> None:
        cfg = PolarizationConfig(**{key: "none"}).model_dump()

        assert cfg[key] is None


class TestSamplingConfig:
    def test_keys(self) -> None:
        cfg = SamplingConfig()
        expected_keys = {
            "mode",
            "device",
            "seed",
            "layout",
            "img_size",
            "fov_center_ra",
            "fov_center_dec",
            "fov_size",
            "corr_int_time",
            "scan_start",
            "scan_duration",
            "num_scans",
            "scan_separation",
            "ref_frequency",
            "frequency_offsets",
            "bandwidths",
            "noisy",
            "corrupted",
            "sensitivity_cut",
        }

        assert set(cfg.model_dump()) == expected_keys

    @pytest.mark.parametrize("mode", ["full", "grid", "dense"])
    def test_valid_mode(self, mode: str) -> None:
        cfg = SamplingConfig(mode=mode)

        assert cfg.mode == mode

    def test_invalid_mode(self) -> None:
        with pytest.raises(ValidationError):
            SamplingConfig(mode="invalid")

    @pytest.mark.parametrize(
        "key,value",
        [
            (key, value)
            for key, values in [
                ("img_size", [0, -1, -64]),
                ("fov_size", [0, -0.1, -10]),
                ("corr_int_time", [0, -1, -100]),
                ("scan_separation", [-1e-5, -10, -1000]),
                ("ref_frequency", [0, -1, -1e8]),
                ("noisy", [-1e-5, -1, -10]),
                ("sensitivity_cut", [-1e-5, -1e8, -1e12]),
            ]
            for value in values
        ],
    )
    def test_invalid_default_field(self, key: str, value: list) -> None:
        with pytest.raises(ValidationError):
            SamplingConfig(**{key: value})

    def test_validate_layout(self) -> None:
        layout = "vla"
        cfg = SamplingConfig(layout=layout)

        assert cfg.layout == layout

    def test_validate_layout_invalid(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            SamplingConfig(layout="invalid_layout")

        assert "expected 'layout' to be one of" in str(excinfo.value)

    def test_validate_dates(self) -> None:
        dates = ["2024-01-01 12:00:00", "2025-01-01 12:00:00"]
        cfg = SamplingConfig(scan_start=dates)

        assert cfg.scan_start == dates

    def test_validate_dates_invalid(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            SamplingConfig(scan_start=["2025-01-01 12:00:00"])

        assert "expected 'scan_start' to be a list of len 2" in str(excinfo.value)

    @pytest.mark.parametrize(
        "seed,expected", [(42, 42), (None, None), ("none", None), (False, None)]
    )
    def test_parse_seed(self, seed, expected) -> None:
        cfg = SamplingConfig(seed=seed)

        assert cfg.seed == expected
