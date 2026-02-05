import importlib
from pathlib import Path

import pytest

import pyvisgen.tools.converter as conv_module

importlib.reload(conv_module)

from pyvisgen.tools.converter import main  # noqa: E402


class TestConverterCLI:
    """Test basic converter functionality."""

    @pytest.mark.parametrize(
        "from_ds,to_ds",
        [
            ("h5", "wds"),
            ("h5", "pt"),
            ("wds", "h5"),
            ("wds", "pt"),
            ("pt", "h5"),
            ("pt", "wds"),
        ],
    )
    def test_calls(
        self, from_ds: str, to_ds: str, tmp_path: Path, mocker, runner
    ) -> None:
        """Test default h5 to wds conversion."""
        mock_instance = mocker.MagicMock()
        mock_dataconverter = mocker.patch("pyvisgen.tools.converter.DataConverter")

        getattr(mock_dataconverter, f"from_{from_ds}").return_value = mock_instance

        result = runner.invoke(
            main,
            [
                str(tmp_path),
                "--input-format",
                from_ds,
                "--output-format",
                to_ds,
            ],
        )

        assert result.exit_code == 0
        mock_dataconverter.assert_called_once()

    @pytest.mark.parametrize(
        "from_ds,to_ds",
        [
            ("h5", "h5"),
            ("wds", "wds"),
            ("pt", "pt"),
        ],
    )
    def test_calls_same_format_raises(
        self, from_ds: str, to_ds: str, tmp_path: Path, mocker, runner
    ) -> None:
        """Test default h5 to wds conversion."""
        mock_instance = mocker.MagicMock()
        mock_dataconverter = mocker.patch("pyvisgen.tools.converter.DataConverter")

        getattr(mock_dataconverter, f"from_{from_ds}").return_value = mock_instance

        result = runner.invoke(
            main,
            [
                str(tmp_path),
                "--input-format",
                from_ds,
                "--output-format",
                to_ds,
            ],
        )

        assert result.exit_code == 2

    def test_dataset_split_str(self, tmp_path: Path, mocker) -> None:
        mock_instance = mocker.MagicMock()
        mock_dataconverter = mocker.patch("pyvisgen.tools.converter.DataConverter")

        mock_dataconverter.from_h5.return_value = mock_instance

        main.callback(
            input_dir=str(tmp_path),
            output_dir=None,
            input_format="h5",
            output_format="wds",
            dataset_split="all",
            amp_phase=True,
            shard_pattern="",
            compress=False,
            bundle_size=100,
        )  # ty:ignore[call-non-callable]

        assert mock_dataconverter.called
