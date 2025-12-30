import re
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
from rich.progress import track

from .datawriters import H5Writer, WDSShardWriter

try:
    import pyarrow as pa
    import webdataset as wds

    _WDS_AVAIL = True
except ImportError:
    _WDS_AVAIL = False


class DataConverter:
    @classmethod
    def from_wds(cls, data_dir, dataset_type="all"):
        if not _WDS_AVAIL:
            raise ImportError(
                "Could not import webdataset. Please make sure you install "
                "pyvisgen with the webdataset extra: "
                "uv pip install pyvisgen[webdataset]"
            )

        cls = cls()
        cls._FMT = "wds"

        cls.datasets = {dataset_type: Path(data_dir).glob(f"{dataset_type}-*.tar*")}
        if dataset_type == "all":
            cls.datasets = {
                "train": Path(data_dir).glob("train-*.tar*"),
                "valid": Path(data_dir).glob("valid-*.tar*"),
                "test": Path(data_dir).glob("test-*.tar*"),
            }

        return cls

    @classmethod
    def from_h5(cls, data_dir, dataset_type="all"):
        cls = cls()
        cls._FMT = "h5"

        cls.datasets = {dataset_type: Path(data_dir).glob(f"*{dataset_type}_*.h5")}
        if dataset_type == "all":
            cls.datasets = {
                "train": Path(data_dir).glob("*train_*.h5"),
                "valid": Path(data_dir).glob("*valid_*.h5"),
                "test": Path(data_dir).glob("*test_*.h5"),
            }

        return cls

    @classmethod
    def from_pt(cls, data_dir, dataset_type="all"):
        raise NotImplementedError("PT will be supported in a future release.")

    def _to_h5(self):
        if self._FMT == "wds":
            for dataset_type, files in track(
                self.datasets.items(), description="Converting Dataset to HDF5"
            ):
                with H5Writer(
                    output_path=self.output_dir,
                    dataset_type=dataset_type,
                    half_image=False,
                ) as writer:
                    for file in track(list(files), description="Processing files..."):
                        file_idx = re.findall(r"\d+", file.stem)
                        file_idx = re.sub(r"0+(.+)", r"\1", *file_idx)

                        webdataset = (
                            wds.WebDataset(str(file), shardshuffle=False)
                            .decode()
                            .to_tuple("input.npy", "target.npy")
                        )

                        x_data = []
                        y_data = []
                        for inp, tar in webdataset:
                            x_data.append(inp)
                            y_data.append(tar)

                        x_data = np.asarray(x_data)
                        y_data = np.asarray(y_data)

                        writer.write(x_data, y_data, index=file_idx)
        elif self._FMT == "pt":
            raise NotImplementedError("PT will be supported in a future release.")
        elif self._FMT == "h5":
            raise RuntimeError("Forbidden: Cannot convert HDF5 to  HDF5.")

    def _to_wds(self):
        if self._FMT == "h5":
            for dataset_type, files in track(
                self.datasets.items(), description="Converting Dataset to HDF5"
            ):
                # total_samples is updated after writing all files
                total_samples = 0

                with WDSShardWriter(
                    output_path=self.output_dir,
                    dataset_type=dataset_type,
                    total_samples=total_samples,
                    shard_pattern=self.shard_pattern,
                    amp_phase=self.amp_phase,
                    compress=self.compress,
                    half_image=False,
                ) as writer:
                    for file in track(list(files), description="Processing files..."):
                        print(file)
                        data = h5py.File(file)
                        file_idx = re.findall(r"\d+", file.stem)

                        writer.write(data["x"], data["y"], index=int(file_idx[0]))
                        total_samples += len(data["x"])

                    for file in self.output_dir.glob(f"{dataset_type}*.parquet"):
                        metadata = pa.parquet.read_table(file).to_pandas()
                        metadata["total_samples_in_dataset"] = total_samples
                        table = pa.Table.from_pandas(metadata)
                        pa.parquet.write_table(table, file)

        elif self._FMT == "pt":
            raise NotImplementedError("PT will be supported in a future release.")
        elif self._FMT == "wds":
            raise RuntimeError("Forbidden: Cannot convert WebDataset to WebDataset.")

    def _to_pt(self):
        pass

    def to(
        self,
        output_dir,
        output_format: str = "h5",
        compress=True,
        shard_pattern="%06d.tar",
        amp_phase=True,
    ):
        self.output_dir = Path(output_dir)
        if not self.output_dir.is_dir():
            self.output_dir.mkdir(parents=True)

        self.compress = compress
        self.shard_pattern = shard_pattern
        self.amp_phase = amp_phase

        match output_format:
            case "h5":
                self._to_h5()
            case "wds":
                self._to_wds()
            case "pt":
                raise NotImplementedError("PT will be supported in a future release.")
