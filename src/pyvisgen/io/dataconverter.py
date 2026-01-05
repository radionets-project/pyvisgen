import re
from pathlib import Path
from typing import Self

import h5py
import numpy as np
import pyarrow as pa
import torch
from natsort import natsorted
from rich.progress import track

from .datawriters import H5Writer, PTWriter, WDSShardWriter

try:
    import pyarrow as pa
    import webdataset as wds

    _WDS_AVAIL = True
except ImportError:
    _WDS_AVAIL = False


__all__ = ["DataConverter"]


def _batch_array(
    array: np.ndarray, batch_size: int, return_indices: bool = False
) -> list[np.ndarray]:
    """Splits array into batches of given batch size. Depending on the batch
    size, the last array may contain the remainder of elements and may
    be smaller batch_size.

    Parameters
    ----------
    array : np.ndarray
        Array to be batched.
    batch_size : int
        Batch size for the splits.
    return_indices : bool, optional
        If ``True``, return indices of splits. Default: ``False``

    Returns
    -------
    list
        List of batched arrays.
    indices
        Indices of splits if return_indices is ``True``.
    """
    indices = np.arange(batch_size, len(array), batch_size)

    if return_indices:
        # also include zero when returning indices
        return np.split(array, indices), np.insert(indices, 0, 0)

    return np.split(array, indices)


class DataConverter:
    """Convert datasets between HDF5, WebDataset, and PyTorch formats.

    This class allows loading datasets from various formats
    and convert them to a target format. Where available or required,
    metadata is read or added to the respective datasets.


    Examples
    --------
    Convert WebDataset to HDF5:

    >>> converter = DataConverter.from_wds("./data/visibilities")
    >>> converter.to("./data/output", output_format="h5")

    Convert HDF5 train split to WebDataset:

    >>> converter = DataConverter.from_h5("./data/visibilities", dataset_type="train")
    >>> converter.to("~/data/output", output_format="wds", compress=True)
    """

    @classmethod
    def from_wds(cls, data_dir, dataset_type="all") -> Self:
        """Create a DataConverter instance from WebDataset files.

        Parameters
        ----------
        data_dir : str or :class:`~pathlib.Path`
            Directory containing WebDataset .tar(.gz) files.
        dataset_type :  str or list
            Dataset split to load.  If "all", loads train, valid, and test.
            Default: ``"all"``

        Returns
        -------
        DataConverter
            Configured DataConverter instance with WebDataset source files.

        Raises
        ------
        ImportError
            If webdataset package is not installed.
        """
        if not _WDS_AVAIL:
            raise ImportError(
                "Could not import webdataset. Please make sure you install "
                "pyvisgen with the webdataset extra: "
                "uv pip install pyvisgen[webdataset]"
            )

        cls = cls()
        cls._FMT = "wds"

        data_dir = Path(data_dir).expanduser().resolve()

        if not isinstance(dataset_type, list):
            dataset_type = [dataset_type]

        if "all" in dataset_type:
            dataset_type = ["train", "valid", "test"]

        cls.datasets = {t: data_dir.glob(f"{t}-*.tar*") for t in dataset_type}

        return cls

    @classmethod
    def from_h5(cls, data_dir, dataset_type="all") -> Self:
        """Create a DataConverter instance from HDF5 files.

        Parameters
        ----------
        data_dir : str or :class:`~pathlib.Path`
            Directory containing HDF5 files.
        dataset_type :  str or list
            Dataset split to load.  If "all", loads train, valid, and test.
            Default: ``"all"``

        Returns
        -------
        DataConverter
            Configured DataConverter instance with HDF5 source files.
        """
        cls = cls()
        cls._FMT = "h5"

        data_dir = Path(data_dir).expanduser().resolve()

        if not isinstance(dataset_type, list):
            dataset_type = [dataset_type]

        if "all" in dataset_type:
            dataset_type = ["train", "valid", "test"]

        cls.datasets = {t: data_dir.glob(f"*{dataset_type}_*.h5") for t in dataset_type}

        return cls

    @classmethod
    def from_pt(cls, data_dir, dataset_type="all"):
        """Create a DataConverter instance from HDF5 files.

        Parameters
        ----------
        data_dir : str or :class:`~pathlib.Path`
            Directory containing .pt files.
        dataset_type :  str or list
            Dataset split to load.  If "all", loads train, valid, and test.
            Default: ``"all"``

        Returns
        -------
        DataConverter
            Configured DataConverter instance with PyTorch pickle source files.
        """
        cls = cls()
        cls._FMT = "pt"

        data_dir = Path(data_dir).expanduser().resolve()

        if not isinstance(dataset_type, list):
            dataset_type = [dataset_type]

        if "all" in dataset_type:
            dataset_type = ["train", "valid", "test"]

        cls.datasets = {t: data_dir.glob(f"*{dataset_type}_*.pt") for t in dataset_type}

        return cls

    def _to_h5(self) -> None:
        """Internal method to handle conversion to HDF5 files."""
        if self._FMT == "wds":
            for dataset_type, files in track(
                self.datasets.items(), description="Converting Dataset to PT"
            ):
                with H5Writer(
                    output_path=self.output_dir,
                    dataset_type=dataset_type,
                    half_image=False,
                ) as writer:
                    self._handle_wds(files, writer)
        elif self._FMT == "pt":
            for dataset_type, files in track(
                self.datasets.items(), description="Converting Dataset to PT"
            ):
                with H5Writer(
                    output_path=self.output_dir,
                    dataset_type=dataset_type,
                    half_image=False,
                ) as writer:
                    bundles, indices = _batch_array(
                        np.asarray(natsorted(files)),
                        self.bundle_size,
                        return_indices=True,
                    )
                    for bundle, index in track(
                        zip(bundles, indices), description="Processing files..."
                    ):
                        x = []
                        y = []
                        for file in bundle:
                            data = torch.load(file)
                            x.append(data["SIM"])
                            y.append(data["TRUTH"])

                        writer.write(
                            np.asarray(x),
                            np.asarray(y),
                            index=int(index),
                            bundle_length=len(data["SIM"]),
                        )

    def _to_wds(self) -> None:
        """Internal method to handle conversion to WebDataset files."""
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
            amp_phase = None
            for dataset_type, files in track(
                self.datasets.items(), description="Converting Dataset to PT"
            ):
                with WDSShardWriter(
                    output_path=self.output_dir,
                    dataset_type=dataset_type,
                    total_samples=total_samples,
                    shard_pattern=self.shard_pattern,
                    compress=self.compress,
                    half_image=False,
                ) as writer:
                    bundles, indices = _batch_array(
                        np.asarray(natsorted(files)),
                        self.bundle_size,
                        return_indices=True,
                    )
                    for bundle, index in track(
                        zip(bundles, indices), description="Processing files..."
                    ):
                        x = []
                        y = []
                        for file in bundle:
                            data = torch.load(file)
                            x.append(data["SIM"])
                            y.append(data["TRUTH"])
                            if not amp_phase:
                                amp_phase = data["TYPE"]

                        writer.write(
                            np.asarray(x),
                            np.asarray(y),
                            index=int(index),
                            bundle_length=len(data["SIM"]),
                        )
                        total_samples += len(x)

                    for file in self.output_dir.glob(f"{dataset_type}*.parquet"):
                        metadata = pa.parquet.read_table(file).to_pandas()
                        metadata["total_samples_in_dataset"] = [total_samples]
                        metadata["data_type"] = [amp_phase]
                        table = pa.Table.from_pandas(metadata)
                        pa.parquet.write_table(table, file)

    def _to_pt(self):
        """Internal method to handle conversion to PT files."""
        if self._FMT == "wds":
            for dataset_type, files in track(
                self.datasets.items(), description="Converting Dataset to PT"
            ):
                with PTWriter(
                    output_path=self.output_dir,
                    dataset_type=dataset_type,
                    amp_phase=self.amp_phase,
                    half_image=False,
                ) as writer:
                    self._handle_wds(files, writer)
        elif self._FMT == "h5":
            for dataset_type, files in track(
                self.datasets.items(), description="Converting Dataset to PT"
            ):
                with PTWriter(
                    output_path=self.output_dir,
                    dataset_type=dataset_type,
                    amp_phase=self.amp_phase,
                    half_image=False,
                ) as writer:
                    for file in track(list(files), description="Processing files..."):
                        data = h5py.File(file)
                        file_idx = re.findall(r"\d+", file.stem)

                        writer.write(
                            data["x"],
                            data["y"],
                            index=int(file_idx[0]),
                            bundle_length=len(data["x"]),
                        )

    def _handle_wds(self, files, writer):
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

            writer.write(x_data, y_data, index=file_idx, bundle_length=len(x_data))

    def to(
        self,
        output_dir: str | Path,
        output_format: str = "h5",
        amp_phase: bool = True,
        shard_pattern: str = "%06d.tar",
        compress: bool = True,
        bundle_size: int = 100,
    ) -> None:
        """Convert the loaded dataset to the specified output format.

        Parameters
        ----------
        output_dir : str or :class:`~pathlib.Path`
            Directory to write converted files to.
        output_format : str
            Target format for conversion. One of h5, wds or pt.
            Default: ``"h5"``
        amp_phase : bool
            Whether to store data in amplitude/phase or real/imaginary
            representation. Default: ``True``
        shard_pattern :  str
            Naming pattern for WebDataset shards (only applies to wds output).
            Default: ``"%06d.tar"``
        compress : bool
            Whether to compress WebDataset shards (only applies to wds output).
            Default: ``True``

        Raises
        ------
        RuntimeError
            If source and target formats are identical.
        """
        if self._FMT.lower() == output_format.lower():
            raise RuntimeError(
                f"Forbidden: Cannot convert {self._FMT} to {output_format}. "
                "Check that input and output formats are different."
            )

        self.output_dir = Path(output_dir).expanduser().resolve()
        if not self.output_dir.is_dir():
            self.output_dir.mkdir(parents=True)

        self.amp_phase = amp_phase
        self.shard_pattern = shard_pattern
        self.compress = compress
        self.bundle_size = bundle_size

        match output_format:
            case "h5":
                self._to_h5()
            case "wds":
                self._to_wds()
            case "pt":
                self._to_pt()
