from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from natsort import natsorted

from pyvisgen.io.datawriters import (
    DataWriter,
    FITSWriter,
    H5Writer,
    PTWriter,
    UVH5Writer,
    WDSShardWriter,
)

try:
    import pyarrow as pa
    import webdataset as wds

    _WDS_AVAIL = True
except ImportError:
    _WDS_AVAIL = False


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
            assert {"x", "y"} == set(f)

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

    def test_arrays_equal(
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
            half_image = x.shape[-2] // 2 + overlap

            np.testing.assert_array_equal(f["x"], x[..., :half_image, :])
            np.testing.assert_array_equal(f["y"], y[..., :half_image, :])


@pytest.mark.skipif(not _WDS_AVAIL, reason="WebDataset is not installed")
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
        expected_type = "amp_phase" if amp_phase else "real_imag"
        assert table["data_type"][0].as_py() == expected_type

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

    def test_arrays_equal(
        self, output_dir: Path, data_sample: tuple[np.ndarray, np.ndarray]
    ) -> None:
        x, y = data_sample

        with WDSShardWriter(
            output_path=output_dir,
            dataset_type="train",
            total_samples=10,
            shard_pattern="%06d.tar",
            amp_phase=True,
        ) as writer:
            writer.write(x, y, index=0)

        webdataset = (
            wds.WebDataset(str(output_dir / "train-000000.tar"), shardshuffle=False)
            .decode()
            .to_tuple("input.npy", "target.npy")
        )

        wds_x, wds_y = next(iter(webdataset))

        overlap = 5
        half_image = x.shape[-2] // 2 + overlap

        np.testing.assert_array_equal(wds_x, x[0, :, :half_image, :])
        np.testing.assert_array_equal(wds_y, y[0, :, :half_image, :])


class TestPTWriter:
    @pytest.mark.parametrize("amp_phase", [True, False])
    def test_write(
        self,
        amp_phase: bool,
        output_dir: Path,
        data_sample: tuple[np.ndarray, np.ndarray],
    ) -> None:
        x, y = data_sample
        bundle_length = len(x)

        with PTWriter(
            output_path=output_dir,
            dataset_type="train",
            total_samples=10,
            amp_phase=amp_phase,
        ) as writer:
            writer.write(x, y, index=0, bundle_length=bundle_length)

        expected_type = "amp_phase" if amp_phase else "real_imag"

        output_files = natsorted(output_dir.glob("*.pt"))
        assert len(output_files) == bundle_length

        for file in output_files:
            data = torch.load(file, weights_only=False)

            assert {"SIM", "TRUTH", "TYPE"} == set(data)
            assert data["TYPE"] == expected_type
            assert data["SIM"].is_sparse
            assert not data["TRUTH"].is_sparse

            assert data["SIM"].to_dense().is_complex
            assert data["TRUTH"].is_complex

    def test_arrays_equal(
        self,
        output_dir: Path,
        data_sample: tuple[np.ndarray, np.ndarray],
    ) -> None:
        x, y = data_sample
        bundle_length = len(x)

        with PTWriter(
            output_path=output_dir,
            dataset_type="train",
            total_samples=10,
            amp_phase=True,
        ) as writer:
            writer.write(x, y, index=0, bundle_length=bundle_length)

        output_files = natsorted(output_dir.glob("*.pt"))
        assert len(output_files) == bundle_length

        overlap = 5
        half_image = x.shape[-2] // 2 + overlap

        for i, file in enumerate(output_files):
            data = torch.load(file, weights_only=False)

            x_dense = data["SIM"].to_dense()
            pt_x = torch.stack((x_dense.real, x_dense.imag), dim=0)
            pt_y = torch.stack((data["TRUTH"].real, data["TRUTH"].imag), dim=0)

            np.testing.assert_array_equal(pt_x, x[i, :, :half_image, :])
            np.testing.assert_array_equal(pt_y, y[i, :, :half_image, :])


class TestUVH5Writer:
    def test_write(self, output_dir, uvh5_vis_data, uvh5_obs) -> None:
        with UVH5Writer(output_path=output_dir, dataset_type="train") as writer:
            writer.write(uvh5_vis_data, uvh5_obs, index=0)

        assert (output_dir / "train_0.uvh5").exists()

    def test_file_groups(self, output_dir, uvh5_vis_data, uvh5_obs) -> None:
        with UVH5Writer(output_path=output_dir, dataset_type="train") as writer:
            writer.write(uvh5_vis_data, uvh5_obs, index=0)

        with h5py.File(output_dir / "train_0.uvh5", "r") as f:
            assert {"visibilities", "uvw", "lmn", "frequency_bands"} <= set(f)

    def test_vis_datasets(self, output_dir, uvh5_vis_data, uvh5_obs) -> None:
        with UVH5Writer(output_path=output_dir, dataset_type="train") as writer:
            writer.write(uvh5_vis_data, uvh5_obs, index=0)

        with h5py.File(output_dir / "train_0.uvh5", "r") as f:
            assert {"V_11", "V_22", "V_12", "V_21", "weights"} == set(f["visibilities"])

    def test_uvw_datasets(self, output_dir, uvh5_vis_data, uvh5_obs) -> None:
        with UVH5Writer(output_path=output_dir, dataset_type="train") as writer:
            writer.write(uvh5_vis_data, uvh5_obs, index=0)

        with h5py.File(output_dir / "train_0.uvh5", "r") as f:
            assert {"u", "v", "w"} == set(f["uvw"])

    def test_lmn_datasets(self, output_dir, uvh5_vis_data, uvh5_obs) -> None:
        with UVH5Writer(output_path=output_dir, dataset_type="train") as writer:
            writer.write(uvh5_vis_data, uvh5_obs, index=0)

        with h5py.File(output_dir / "train_0.uvh5", "r") as f:
            assert {"l", "m", "n"} == set(f["lmn"])

    def test_frequency_bands(self, output_dir, uvh5_vis_data, uvh5_obs) -> None:
        with UVH5Writer(output_path=output_dir, dataset_type="train") as writer:
            writer.write(uvh5_vis_data, uvh5_obs, index=0)

        expected = (uvh5_obs.ref_frequency + uvh5_obs.frequency_offsets).numpy()

        with h5py.File(output_dir / "train_0.uvh5", "r") as f:
            np.testing.assert_array_almost_equal(f["frequency_bands"][...], expected)

    def test_sky_written(self, output_dir, uvh5_vis_data, uvh5_obs, uvh5_sky) -> None:
        with UVH5Writer(output_path=output_dir, dataset_type="train") as writer:
            writer.write(uvh5_vis_data, uvh5_obs, index=0, sky=uvh5_sky)

        with h5py.File(output_dir / "train_0.uvh5", "r") as f:
            assert "sky" in f
            assert "SI" in f["sky"]
            np.testing.assert_array_equal(f["sky/SI"][...], uvh5_sky.numpy())

    def test_sky_not_written_when_none(
        self, output_dir, uvh5_vis_data, uvh5_obs
    ) -> None:
        with UVH5Writer(output_path=output_dir, dataset_type="train") as writer:
            writer.write(uvh5_vis_data, uvh5_obs, index=0, sky=None)

        with h5py.File(output_dir / "train_0.uvh5", "r") as f:
            assert "sky" not in f

    def test_arrays_equal(self, output_dir, uvh5_vis_data, uvh5_obs) -> None:
        with UVH5Writer(output_path=output_dir, dataset_type="train") as writer:
            writer.write(uvh5_vis_data, uvh5_obs, index=0)

        with h5py.File(output_dir / "train_0.uvh5", "r") as f:
            np.testing.assert_array_equal(
                f["visibilities/V_11"][...], uvh5_vis_data.V_11.numpy()
            )
            np.testing.assert_array_equal(
                f["visibilities/weights"][...], uvh5_vis_data.weights.numpy()
            )
            np.testing.assert_array_equal(f["uvw/u"][...], uvh5_vis_data.u.numpy())
            np.testing.assert_array_equal(f["uvw/v"][...], uvh5_vis_data.v.numpy())
            np.testing.assert_array_equal(f["uvw/w"][...], uvh5_vis_data.w.numpy())

    def test_lmn_values(self, output_dir, uvh5_vis_data, uvh5_obs) -> None:
        with UVH5Writer(output_path=output_dir, dataset_type="train") as writer:
            writer.write(uvh5_vis_data, uvh5_obs, index=0)

        lm = uvh5_obs.lm.numpy()
        expected_n = np.sqrt(np.maximum(1.0 - lm[..., 0] ** 2 - lm[..., 1] ** 2, 0.0))

        with h5py.File(output_dir / "train_0.uvh5", "r") as f:
            np.testing.assert_array_almost_equal(f["lmn/l"][...], lm[..., 0])
            np.testing.assert_array_almost_equal(f["lmn/m"][...], lm[..., 1])
            np.testing.assert_array_almost_equal(f["lmn/n"][...], expected_n)

    def test_write_multiple_samples(self, output_dir, uvh5_vis_data, uvh5_obs) -> None:
        with UVH5Writer(output_path=output_dir, dataset_type="train") as writer:
            for i in range(3):
                writer.write(uvh5_vis_data, uvh5_obs, index=i)

        files = list(output_dir.glob("*.uvh5"))
        assert len(files) == 3
        assert {f.name for f in files} == {
            "train_0.uvh5",
            "train_1.uvh5",
            "train_2.uvh5",
        }

    def test_filename_pattern(self, output_dir, uvh5_vis_data, uvh5_obs) -> None:
        with UVH5Writer(output_path=output_dir, dataset_type="valid") as writer:
            writer.write(uvh5_vis_data, uvh5_obs, index=5)

        assert (output_dir / "valid_5.uvh5").exists()


class TestFITSWriter:
    def test_write(self, output_dir: Path, mocker) -> None:
        vis_data = None
        obs = None

        mock_create_hdu_list = mocker.patch("pyvisgen.io.datawriters.create_hdu_list")

        writer = FITSWriter(output_path=output_dir, dataset_type="train")
        writer.write(vis_data, obs, index=0)

        # check that create_hdu_list is called
        # but do not call actual method since we're
        # missing vis_data and obs here
        assert mock_create_hdu_list.called


class TestWriterABC:
    @pytest.fixture
    def mock_writer(self):
        class _MockWriter(DataWriter):
            def __init__(self, output_path: Path, dataset_type: str) -> None:
                pass

            def write(self) -> None:
                pass

        return _MockWriter

    def test_abstractmethod_raises(self) -> None:
        with pytest.raises(TypeError) as excinfo:
            DataWriter(".", dataset_type="train")

        assert "Can't instantiate abstract class DataWriter " in str(excinfo.value)

    def test_test_shapes(
        self, mock_writer, data_sample: tuple[np.ndarray, np.ndarray]
    ) -> None:
        writer = mock_writer(
            ".", dataset_type="train"
        )  # should not raise any exception

        x, y = data_sample

        writer.test_shapes(x, "x")
        writer.test_shapes(y, "y")

    @pytest.mark.parametrize(
        "shape,expected_errmsg",
        [
            ((10, 3, 32, 32), "Expected array test_arr axis 1 to be 2"),
            ((3, 32, 32), "Expected array test_arr axis 1 to be 2"),
            ((10, 2, 32), "Expected array test_arr ndim to be 4"),
            ((5, 2, 10, 32, 32), "Expected array test_arr ndim to be 4"),
        ],
    )
    def test_test_shapes_invalid(
        self,
        shape: tuple,
        expected_errmsg: str,
        mock_writer,
    ) -> None:
        writer = mock_writer(".", dataset_type="train")

        rng = np.random.default_rng()
        arr = rng.uniform(size=shape)

        with pytest.raises(ValueError) as excinfo:
            writer.test_shapes(arr, "test_arr")

        assert expected_errmsg in str(excinfo.value)

    @pytest.mark.parametrize("overlap", [1, 2, 3, 4, 5])
    def test_get_half_image(
        self, overlap: int, mock_writer, data_sample: tuple[np.ndarray, np.ndarray]
    ) -> None:
        writer = mock_writer(".", dataset_type="train")

        x, y = data_sample
        half_x, half_y = writer.get_half_image(x, y, overlap=overlap)

        half_idx = x.shape[-2] // 2 + overlap
        np.testing.assert_array_equal(half_x, x[..., :half_idx, :])
        np.testing.assert_array_equal(half_y, y[..., :half_idx, :])
