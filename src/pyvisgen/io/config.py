import inspect
import os
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from pyvisgen.io import datawriters
from pyvisgen.layouts import get_array_names

__all__ = [
    "Config",
    "SamplingConfig",
    "PolarizationConfig",
    "BundleConfig",
]


class SamplingConfig(BaseModel, validate_assignment=True):
    """Sampling config BaseModel"""

    mode: Literal["full", "grid", "dense"] = "full"
    device: str = "cuda"
    seed: str | bool | int | None = 1337
    layout: str = "vlba"
    img_size: int = Field(default=1024, gt=0)
    fov_center_ra: list[int] = [100, 110]
    fov_center_dec: list[int] = [30, 40]
    fov_size: float = Field(default=0.24, gt=0)
    corr_int_time: float = Field(default=30.0, gt=0)
    scan_start: list[str] = ["01-01-1995 00:00:01", "01-01-2025 23:59:59"]
    scan_duration: list[int] = [20, 600]
    num_scans: list[int] = [6, 10]
    scan_separation: float = Field(default=360, ge=0)
    ref_frequency: float = Field(default=15.17600e9, gt=0)
    frequency_offsets: list[float] = [0e8, 1.28e8, 2.56e8, 3.84e8]
    bandwidths: list[float] = [1.28e8, 1.28e8, 1.28e8, 1.28e8]
    noisy: int = Field(default=0, ge=0)
    corrupted: bool = False
    sensitivity_cut: float = Field(default=1e-6, ge=0)

    @field_validator("layout")
    @classmethod
    def validate_layout(cls, layout: str) -> None:
        _avail_layouts = get_array_names()

        if layout not in _avail_layouts:
            raise ValueError(
                f"expected 'layout' to be one of {_avail_layouts} but got {layout}"
            )
        return layout

    @field_validator("scan_start")
    @classmethod
    def validate_dates(cls, v: list[str]) -> None:
        if len(v) != 2:
            raise ValueError("expected 'scan_start' to be a list of len 2")

        return v

    @field_validator("seed")
    @classmethod
    def parse_seed(cls, v: str | bool | int | None) -> int | None:
        if v in {"none", False}:
            v = None

        return v


class PolarizationConfig(BaseModel, validate_assignment=True):
    """Polarization config BaseModel"""

    mode: Literal["linear", "circular", "none"] | None = None
    delta: float | None = Field(default=45)
    amp_ratio: Annotated[float, Field(ge=0.0, le=1.0)] | None = Field(default=0.5)
    field_order: list[float] | None = [0.01, 0.01]
    field_scale: list[float] | None = [0.0, 1.0]
    field_threshold: float | None = None

    @field_validator(
        "mode",
        "delta",
        "amp_ratio",
        "field_order",
        "field_scale",
        "field_threshold",
        mode="before",
    )
    @classmethod
    def parse_mode_thresh(
        cls, v: str | float | list | None
    ) -> str | float | list | None:
        if v == "none":
            v = None

        return v


class BundleConfig(BaseModel, validate_assignment=True):
    """Bundle config BaseModel"""

    dataset_type: Literal["train", "test", "valid", "none", ""] = "train"
    in_path: str | Path = "./path/to/input/data/"
    out_path: str | Path = "./output/path/"
    overlap: int = 5
    grid_size: int = Field(default=1024, gt=0)
    grid_fov: float = Field(default=0.24, gt=0)
    amp_phase: bool = False

    @field_validator("in_path", "out_path")
    @classmethod
    def expand_path(cls, v: str | Path, info: ValidationInfo) -> Path:
        """Expand and resolve paths."""

        if v in {"none", ""}:
            raise ValueError(f"'{info.field_name}' cannot be empty!")

        v = Path(v).expanduser().resolve()

        return v


class DataWriterConfig(BaseModel, validate_assignment=True):
    writer: str | Callable = datawriters.H5Writer
    overlap: int = Field(default=5, gt=0)
    shard_pattern: str = "%06d.tar"
    compress: bool = False

    @field_validator("writer")
    @classmethod
    def setup_writer(cls, writer) -> Callable:
        if isinstance(writer, Callable) and issubclass(writer, datawriters.DataWriter):
            return writer

        _avail_writers = {}

        for member in inspect.getmembers(datawriters):
            if inspect.isclass(member[1]):
                _avail_writers[member[0]] = member[1]

        # handle shorthands for full data writer names
        if writer.lower() in ["h5", "hdf5"]:
            output_writer = _avail_writers["H5Writer"]
        elif writer.lower() in ["wds", "webdataset"]:
            output_writer = _avail_writers["WDSShardWriter"]
        elif writer.lower() in ["pt"]:
            output_writer = _avail_writers["PTWriter"]
        else:
            output_writer = _avail_writers[writer]

        return output_writer


class GriddingConfig(BaseModel, validate_assignment=True):
    gridder: str = "default"


class FFTConfig(BaseModel, validate_assignment=True):
    ft: Literal["default", "finufft", "reversed"] = "default"


class CodeCarbonEmissionTrackerConfig(BaseModel, validate_assignment=True):
    """Codecarbon emission tracker configuration"""

    log_level: str | int = "error"
    country_iso_code: str = "DEU"
    output_dir: str | None = os.getcwd()


class Config(BaseModel):
    """Main training configuration."""

    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    polarization: PolarizationConfig = Field(default_factory=PolarizationConfig)
    bundle: BundleConfig = Field(default_factory=BundleConfig)
    datawriter: DataWriterConfig = Field(default_factory=DataWriterConfig)
    gridding: GriddingConfig = Field(default_factory=GriddingConfig)
    fft: FFTConfig = Field(default_factory=FFTConfig)
    codecarbon: bool | CodeCarbonEmissionTrackerConfig = False

    @classmethod
    def from_toml(cls, path: str | Path) -> "Config":
        """Load configuration from a TOML file."""
        with open(path, "rb") as f:
            data = tomllib.load(f)

        return cls(**data)

    @field_validator("codecarbon", mode="after")
    @classmethod
    def validate_codecarbon(cls, v: bool | CodeCarbonEmissionTrackerConfig):
        if isinstance(v, dict):  # pragma: no cover
            return CodeCarbonEmissionTrackerConfig(**v, project_name="pyvisgen")
        elif v is True:
            return CodeCarbonEmissionTrackerConfig(project_name="pyvisgen")

        return v
