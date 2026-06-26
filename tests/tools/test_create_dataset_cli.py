import importlib

import pytest

import pyvisgen.tools.create_dataset as conv_module

# reload converter module so that we get a fresh import of
# the pyvisgen.io.dataconvert.DataConverter class
# which for some reason fails the following tests if the
# tests for DataConverter run before them
importlib.reload(conv_module)

from pyvisgen.tools.create_dataset import main  # noqa: E402


class TestCreateDatasetCLI:
    def test_cli_help(self, runner) -> None:
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0

    @pytest.mark.parametrize("mode", ["simulate", "slurm", "gridding"])
    def test_modes(self, mode, mocker, runner) -> None:
        mock_instance = mocker.MagicMock()
        mock_simulate_dataset = mocker.patch(
            "pyvisgen.tools.create_dataset.SimulateDataSet",
        )

        mock_simulate_dataset.from_config.return_value = mock_instance

        result = runner.invoke(main, ["--mode", mode, "tests/test_conf.toml"])

        assert result.exit_code == 0
        mock_simulate_dataset.from_config.assert_called_once()
