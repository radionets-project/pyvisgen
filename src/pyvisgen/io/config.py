import inspect
import tomllib
from pathlib import Path
from typing import Callable, Literal

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
    seed: int | None = 1337
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


class PolarizationConfig(BaseModel, validate_assignment=True):
    """Polarization config BaseModel"""

    mode: Literal["linear", "circular", "none"] | None = None
    delta: float = Field(default=45)
    amp_ratio: float = Field(default=0.5)
    field_order: list[float] = [0.01, 0.01]
    field_scale: list[float] = [0, 1]
    field_threshold: float | Literal["none"] | None = None

    @field_validator("mode", "field_threshold")
    @classmethod
    def parse_mode_thresh(cls, v: str | float) -> str | float | None:
        if v == "none":
            v = None

        return v


class BundleConfig(BaseModel, validate_assignment=True):
    """Bundle config BaseModel"""

    dataset_type: Literal["train", "test", "valid", "none", ""] = "train"
    in_path: str | Path = "/path/to/input/data/"
    out_path: str | Path = "/output/path/"
    output_writer: str | Callable = datawriters.H5Writer
    grid_size: int = Field(default=1024, gt=0)
    grid_fov: float = Field(default=0.24, gt=0)
    amp_phase: bool = False

    @field_validator("in_path", "out_path")
    @classmethod
    def expand_path(cls, v: Path, info: ValidationInfo) -> Path:
        """Expand and resolve paths."""

        if v in {None, False, "none", ""}:
            raise ValueError(f"'{info.field_name}' cannot be empty!")
        else:
            v = Path(v)
            v.expanduser().resolve()

        return v

    @field_validator("output_writer")
    @classmethod
    def setup_output_writer(cls, output_writer) -> Callable:
        if isinstance(output_writer, Callable) and issubclass(
            output_writer, datawriters.DataWriter
        ):
            return output_writer

        _avail_writers = {}

        for member in inspect.getmembers(datawriters):
            if inspect.isclass(member[1]):
                _avail_writers[member[0]] = member[1]

        if output_writer.lower() in ["h5", "hdf5"] or output_writer == "H5Writer":
            writer = _avail_writers["H5Writer"]

        elif output_writer in ["wds", "webdataset"]:
            raise NotImplementedError(
                "The WebDataset functionality will be implemented in a future release "
                "of pyvisgen."
            )

        return writer


class Config(BaseModel):
    """Main training configuration."""

    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    polarization: PolarizationConfig = Field(default_factory=PolarizationConfig)
    bundle: BundleConfig = Field(default_factory=BundleConfig)
    # codecarbon: bool | CodeCarbonEmissionTrackerConfig = False

    @classmethod
    def from_toml(cls, path: str | Path) -> "Config":
        """Load configuration from a TOML file."""
        with open(path, "rb") as f:
            data = tomllib.load(f)

        return cls(**data)

    def to_dict(self) -> dict:
        """Export configuration as a dictionary."""
        return self.model_dump()

    # @field_validator("codecarbon", mode="after")
    # @classmethod
    # def validate_codecarbon(cls, v: bool | CodeCarbonEmissionTrackerConfig):
    #     if isinstance(v, dict):
    #         return CodeCarbonEmissionTrackerConfig(
    #             **v, project_name=cls.logging.project_name
    #         )
    #     elif v is True:
    #         return CodeCarbonEmissionTrackerConfig(
    #             project_name=cls.logging.project_name
    #         )
    #
    #     return v
