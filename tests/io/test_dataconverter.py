from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from pyvisgen.io.dataconverter import DataConverter, DataTypeConverter, _batch_array


class TestBatchArray:
    def test_split_even(self) -> None:
        arr = np.arange(20)
        batches = _batch_array(arr, batch_size=10)

        assert len(batches) == 2
        np.testing.assert_array_equal(batches[0], np.arange(10))
        np.testing.assert_array_equal(batches[1], np.arange(10, 20))

    def test_split_uneven(self) -> None:
        arr = np.arange(21)
        batches = _batch_array(arr, batch_size=10)

        assert len(batches) == 3
        assert len(batches[-1]) == 1
        np.testing.assert_array_equal(batches[2], np.arange(20, 21))

    def test_return_indices(self) -> None:
        arr = np.arange(20)
        batches, indices = _batch_array(arr, batch_size=5, return_indices=True)

        assert len(batches) == 4
        np.testing.assert_array_equal(indices, [0, 5, 10, 15])

    def test_batch_larger_than_array(self) -> None:
        arr = np.arange(20)
        batches = _batch_array(arr, batch_size=40)

        assert len(batches) == 1
        np.testing.assert_array_equal(batches[0], arr)


class TestDataConverterFromDataset:
    def test_from_h5(self, h5_dataset: Path) -> None:
        converter = DataConverter.from_h5(h5_dataset, dataset_split="train")

        assert converter._FMT == "h5"
        assert set(converter.datasets) == {"train"}
        assert len(list(converter.datasets["train"])) == 1  # expect 1 file

    def test_from_wds(self, wds_dataset: Path) -> None:
        converter = DataConverter.from_wds(wds_dataset, dataset_split="train")

        assert converter._FMT == "wds"
        assert set(converter.datasets) == {"train"}
        assert len(list(converter.datasets["train"])) == 1  # expect 1 file

    def test_from_pt(self, pt_dataset: Path) -> None:
        converter = DataConverter.from_pt(pt_dataset, dataset_split="train")

        assert converter._FMT == "pt"
        assert set(converter.datasets) == {"train"}

        # expect 4 files since sample_data.shape[0] == 4
        # (each image is saved to an individual file)
        assert len(list(converter.datasets["train"])) == 4

    def test_dataset_split_all(self, h5_dataset: Path) -> None:
        converter = DataConverter.from_h5(h5_dataset, dataset_split="all")

        assert set(converter.datasets) == {"train", "valid", "test"}
        assert len(list(converter.datasets["train"])) == 1
        assert len(list(converter.datasets["valid"])) == 1
        assert len(list(converter.datasets["test"])) == 1

    def test_dataset_split_list(self, h5_dataset: Path) -> None:
        converter = DataConverter.from_h5(h5_dataset, dataset_split=["train", "valid"])

        assert set(converter.datasets) == {"train", "valid"}


class TestDataConverterToDataset:
    def test_same_format_raises(self, tmp_path: Path, h5_dataset: Path) -> None:
        output_dir = tmp_path / "output/"

        converter = DataConverter.from_h5(h5_dataset, dataset_split="train")

        with pytest.raises(RuntimeError) as excinfo:
            converter.to(output_dir, output_format="h5")

        assert "Forbidden: Cannot convert h5 to h5" in str(excinfo.value)

    def test_convert_without_amp_phase(self, tmp_path: Path, h5_dataset: Path):
        output_dir = tmp_path / "output/"

        converter = DataConverter.from_h5(h5_dataset, dataset_split="train")

        with pytest.raises(ValueError) as excinfo:
            converter.to(
                output_dir,
                output_format="h5",
                amp_phase=None,
                convert_representation=True,
            )

        assert "Cannot convert data representation" in str(excinfo.value)

    def test_create_output_dir(self, tmp_path: Path, h5_dataset: Path) -> None:
        output_dir = tmp_path / "output/"

        converter = DataConverter.from_h5(h5_dataset, dataset_split="train")
        converter.to(output_dir, output_format="pt")

        assert output_dir.is_dir()

    @pytest.mark.parametrize("convert", [True, False])
    def test_h5_to_wds(
        self,
        convert: bool,
        tmp_path: Path,
        h5_dataset: Path,
        test_image: tuple[np.ndarray, np.ndarray],
    ) -> None:
        output_dir = tmp_path / "output/"

        converter = DataConverter.from_h5(h5_dataset, dataset_split="train")
        converter.to(output_dir, output_format="wds", convert_representation=convert)

        wds_shards = list(output_dir.glob("*.tar.gz"))
        parquet_files = list(output_dir.glob("*.parquet"))
        assert len(wds_shards) == 1
        assert len(parquet_files) == 1

        if not convert:
            x, y = test_image.get_h5(tmp_path / "samp_train_0.h5")
            x_wds, y_wds, data_type = test_image.get_wds(file=wds_shards[0])

            np.testing.assert_array_equal(x, x_wds)
            np.testing.assert_array_equal(y, y_wds)

            assert data_type == "amp_phase"

    @pytest.mark.parametrize("convert", [True, False])
    def test_h5_to_pt(
        self,
        convert: bool,
        tmp_path: Path,
        h5_dataset: Path,
        test_image: tuple[np.ndarray, np.ndarray],
    ) -> None:
        output_dir = tmp_path / "output/"

        converter = DataConverter.from_h5(h5_dataset, dataset_split="train")
        converter.to(output_dir, output_format="pt", convert_representation=convert)

        pt_files = sorted(output_dir.glob("*.pt"))
        assert len(pt_files) == 4

        if not convert:
            x, y = test_image.get_h5(tmp_path / "samp_train_0.h5")
            x_pt, y_pt, data_type = test_image.get_pt(file=pt_files[0])

            np.testing.assert_array_equal(x, x_pt)
            np.testing.assert_array_equal(y, y_pt)

            assert data_type == "amp_phase"

    @pytest.mark.parametrize("convert", [True, False])
    def test_wds_to_h5(
        self,
        convert: bool,
        tmp_path: Path,
        wds_dataset: Path,
        test_image: tuple[np.ndarray, np.ndarray],
    ) -> None:
        output_dir = tmp_path / "output/"

        converter = DataConverter.from_wds(wds_dataset, dataset_split="train")
        converter.to(output_dir, output_format="h5", convert_representation=convert)

        h5_files = list(output_dir.glob("*.h5"))
        assert len(h5_files) == 1

        if not convert:
            x, y, _ = test_image.get_wds(tmp_path / "train-000000.tar.gz")
            x_h5, y_h5 = test_image.get_h5(h5_files[0])

            np.testing.assert_array_equal(x, x_h5)
            np.testing.assert_array_equal(y, y_h5)

    @pytest.mark.parametrize("convert", [True, False])
    def test_wds_to_pt(
        self,
        convert: bool,
        tmp_path: Path,
        wds_dataset: Path,
        test_image: tuple[np.ndarray, np.ndarray],
    ) -> None:
        output_dir = tmp_path / "output/"

        converter = DataConverter.from_wds(wds_dataset, dataset_split="train")
        converter.to(output_dir, output_format="pt", convert_representation=convert)

        pt_files = sorted(output_dir.glob("*.pt"))
        assert len(pt_files) == 4

        if not convert:
            x, y, data_type = test_image.get_wds(tmp_path / "train-000000.tar.gz")
            x_pt, y_pt, data_type_pt = test_image.get_pt(file=pt_files[0])

            np.testing.assert_array_equal(x, x_pt)
            np.testing.assert_array_equal(y, y_pt)

            assert data_type_pt == data_type  # "amp_phase"

    @pytest.mark.parametrize("convert", [True, False])
    def test_pt_to_h5(
        self,
        convert: bool,
        tmp_path: Path,
        pt_dataset: Path,
        test_image: tuple[np.ndarray, np.ndarray],
    ) -> None:
        output_dir = tmp_path / "output/"

        converter = DataConverter.from_pt(pt_dataset, dataset_split="train")
        converter.to(output_dir, output_format="h5", convert_representation=convert)

        h5_files = list(output_dir.glob("*.h5"))
        assert len(h5_files) == 1

        if not convert:
            x, y, _ = test_image.get_pt(tmp_path / "samp_train_1.pt")
            x_h5, y_h5 = test_image.get_h5(h5_files[0])

            np.testing.assert_array_equal(x, x_h5)
            np.testing.assert_array_equal(y, y_h5)

    @pytest.mark.parametrize("convert", [True, False])
    def test_pt_to_wds(
        self,
        convert: bool,
        tmp_path: Path,
        pt_dataset: Path,
        test_image: tuple[np.ndarray, np.ndarray],
    ) -> None:
        output_dir = tmp_path / "output/"

        converter = DataConverter.from_pt(pt_dataset, dataset_split="train")
        converter.to(output_dir, output_format="wds", convert_representation=convert)

        wds_shards = list(output_dir.glob("*.tar.gz"))
        parquet_files = list(output_dir.glob("*.parquet"))
        assert len(wds_shards) == 1
        assert len(parquet_files) == 1

        if not convert:
            x, y, data_type = test_image.get_pt(tmp_path / "samp_train_1.pt")
            x_wds, y_wds, data_type_wds = test_image.get_wds(file=wds_shards[0])

            np.testing.assert_array_equal(x, x_wds)
            np.testing.assert_array_equal(y, y_wds)

            assert data_type_wds == data_type  # "amp_phase"

    def test_repr_self_convert_h5(self, tmp_path: Path, h5_dataset: Path):
        output_dir = tmp_path / "output/"

        converter = DataConverter.from_h5(h5_dataset, dataset_split="train")
        converter.to(output_dir, output_format="h5", convert_representation=True)

        h5_files = list(output_dir.glob("*.h5"))
        assert len(h5_files) == 1

    def test_repr_self_convert_wds(self, tmp_path: Path, wds_dataset: Path):
        output_dir = tmp_path / "output/"

        converter = DataConverter.from_wds(wds_dataset, dataset_split="train")
        converter.to(output_dir, output_format="wds", convert_representation=True)

        wds_shards = list(output_dir.glob("*.tar.gz"))
        parquet_files = list(output_dir.glob("*.parquet"))
        assert len(wds_shards) == 1
        assert len(parquet_files) == 1

    def test_repr_self_convert_pt(self, tmp_path: Path, pt_dataset: Path):
        output_dir = tmp_path / "output/"

        converter = DataConverter.from_pt(pt_dataset, dataset_split="train")
        converter.to(output_dir, output_format="pt", convert_representation=True)

        pt_files = sorted(output_dir.glob("*.pt"))
        assert len(pt_files) == 4


class TestDataTypeConverter:
    @pytest.fixture
    def data(self):
        x = torch.ones((10, 32, 32))
        y = torch.zeros((10, 32, 32))

        data = torch.stack((x, y), dim=1)

        return data

    def test_to_amp_phase(self, data):
        dtype_conv = DataTypeConverter()

        conv = dtype_conv.to_amp_phase(data)

        assert conv.shape == data.shape
        np.testing.assert_array_equal(conv[:, 0], torch.ones((10, 32, 32)))
        np.testing.assert_array_equal(conv[:, 1], torch.zeros((10, 32, 32)))

    def test_to_real_imag(self, data):
        dtype_conv = DataTypeConverter()

        conv = dtype_conv.to_real_imag(data)

        assert conv.shape == data.shape
        np.testing.assert_array_equal(conv[:, 0], torch.ones((10, 32, 32)))
        np.testing.assert_array_equal(conv[:, 1], torch.zeros((10, 32, 32)))

    @patch.object(DataTypeConverter, "to_amp_phase")
    @patch.object(DataTypeConverter, "to_real_imag")
    def test_convert(self, mock_to_real_imag, mock_to_amp_phase, data):
        DataTypeConverter(input_amp_phase=True).convert(data)
        mock_to_real_imag.assert_called_once_with(data)

        DataTypeConverter(input_amp_phase=False).convert(data)
        mock_to_amp_phase.assert_called_once_with(data)
