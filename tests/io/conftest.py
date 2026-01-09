from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pytest
import torch
import webdataset as wds


@pytest.fixture
def sample_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)

    shape = (4, 2, 16, 32)
    x = rng.random(shape, dtype=np.float32)
    y = rng.random(shape, dtype=np.float32)

    return x, y


@pytest.fixture
def h5_dataset(tmp_path: Path, sample_data: tuple[np.ndarray, np.ndarray]) -> Path:
    x, y = sample_data

    if not tmp_path.is_dir():
        tmp_path.mkdir()

    for split in ["train", "valid", "test"]:
        file = tmp_path / f"samp_{split}_0.h5"

        with h5py.File(file, "w") as f:
            f.create_dataset("x", data=x)
            f.create_dataset("y", data=y)

    return tmp_path


@pytest.fixture
def wds_dataset(tmp_path: Path, sample_data: tuple[np.ndarray, np.ndarray]) -> Path:
    x, y = sample_data

    if not tmp_path.is_dir():
        tmp_path.mkdir()

    for split in ["train", "valid", "test"]:
        shard_path = tmp_path / f"{split}-000000.tar.gz"

        with wds.TarWriter(str(shard_path), compress=True) as tarwriter:
            for i in range(len(x)):
                tarwriter.write({
                    "__key__": f"{split}_{i:08d}",
                    "input.npy": x[i],
                    "target.npy": y[i],
                })

        metadict = {
            "total_samples_in_dataset": [len(x)],
            "samples_in_shard": [len(x)],
            "shard_idx": [0],
            "bundle_id": [0],
            "data_type": ["amp_phase"],
        }
        metadata = pa.Table.from_pydict(metadict)
        metadata_path = (
            str(shard_path).replace(".tar.gz", ".parquet")
        )
        pa.parquet.write_table(metadata, metadata_path)

        return tmp_path


@pytest.fixture
def pt_dataset(tmp_path: Path, sample_data: tuple[np.ndarray, np.ndarray]) -> Path:
    x, y = sample_data

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    x = x[:, 0] + 1j * x[:, 1]
    y = y[:, 0] + 1j * y[:, 1]

    for split in ["train", "valid", "test"]:
        for i in range(len(x)):
            file_path = tmp_path / f"samp_{split}_{i + 1}.pt"

            torch.save(
                obj={"SIM": x[i].to_sparse(), "TRUTH": y[i], "TYPE": "amp_phase"},
                f=file_path,
            )

    return tmp_path


class TestImage:

    @staticmethod
    def get_h5(file: Path) -> tuple[np.ndarray, np.ndarray]:
        with h5py.File(file) as f:
            x = np.asarray(f["x"], dtype=np.float32)[0]
            y = np.asarray(f["y"], dtype=np.float32)[0]

        return x, y

    @staticmethod
    def get_wds(file: Path) -> tuple[np.ndarray, np.ndarray]:
        webdataset = (
            wds.WebDataset(str(file), shardshuffle=False)
            .decode()
            .to_tuple("input.npy", "target.npy")
        )

        x, y = next(iter(webdataset))

        meta_file = (
            str(file).replace(".tar.gz", ".parquet")
            if str(file).endswith(".tar.gz")
            else str(file).replace(".tar", ".parquet")
        )
        metadata = pa.parquet.read_table(meta_file).to_pandas()
        data_type = metadata["data_type"][0]

        return x, y, data_type

    @staticmethod
    def get_pt(file: Path) -> tuple[np.ndarray, np.ndarray]:
        data = torch.load(file)
        x = data["SIM"].to_dense()
        y = data["TRUTH"]
        data_type = data["TYPE"]

        x = np.stack((x.real, x.imag), axis=0)
        y = np.stack((y.real, y.imag), axis=0)

        return x, y, data_type

@pytest.fixture
def test_image():
    return TestImage
