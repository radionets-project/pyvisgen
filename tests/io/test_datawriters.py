from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pytest

from pyvisgen.io import H5Writer, PTWriter, WDSShardWriter


class TestH5Writer:
    def test_write(
        self, output_dir: Path, data_sample: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = data_sample

        with H5Writer(output_path=output_dir, dataset_type="train") as writer:
            writer.write(x, y, index=0)

        output_file = output_dir / "samp_train_0.h5"
        assert output_file.exists()

        with h5py.File(output_file, "r") as f:
            assert "x" in f
            assert "y" in f

            overlap = 5
            expected_h = x.shape[-2] // 2 + overlap
            assert f["x"].shape == (10, 2, expected_h, 32)
            assert f["y"].shape == (10, 2, expected_h, 32)

    def test_write_custom_names(
        self, output_dir: Path, data_sample: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = data_sample

        with H5Writer(output_path=output_dir, dataset_type="train") as writer:
            writer.write(x, y, index=1, name_x="inputs", name_y="targets")

        output_file = output_dir / "samp_train_1.h5"
        with h5py.File(output_file, "r") as f:
            assert "inputs" in f
            assert "targets" in f


class TestWDSShardWriter:
    @pytest.mark.parametrize("amp_phase", [True, False])
    def test_write(
        self,
        amp_phase: bool,
        output_dir: Path,
        data_sample: tuple[np.ndarray, np.ndarray],
    ) -> None:
        x, y = data_sample

        with WDSShardWriter(
            output_path=output_dir,
            dataset_type="train",
            total_samples=10,
            shard_pattern="%06d.tar",
            amp_phase=amp_phase,
        ) as writer:
            writer.write(x, y, index=0)

        output_file = output_dir / "train-000000.tar"
        assert output_file.exists()

        parquet_file = output_dir / "train-000000.parquet"
        assert parquet_file.exists()

        table = pa.parquet.read_table(parquet_file)
        expected = "amp_phase" if amp_phase else "real_imag"
        assert table["data_type"][0].as_py() == expected

    def test_path(self, output_dir: Path) -> None:
        writer = WDSShardWriter(
            output_path=str(output_dir),  # should be converted to Path
            dataset_type="train",
            total_samples=10,
            shard_pattern="%06d.tar",
            amp_phase=True,
        )

        assert isinstance(writer.output_path, Path)

    @pytest.mark.parametrize(
        "compress,shard_pattern",
        [
            (True, "%06d.tar.gz"),
            (True, "%06d.tar"),
            (False, "%06d.tar.gz"),
        ],
    )
    def test_compression(
        self, compress: bool, shard_pattern: str, output_dir: Path
    ) -> None:
        writer = WDSShardWriter(
            output_path=output_dir,
            dataset_type="train",
            total_samples=10,
            shard_pattern=shard_pattern,
            amp_phase=True,
            compress=compress,
        )

        # should have .tar.gz suffix
        assert writer.shard_pattern == "%06d.tar.gz"

