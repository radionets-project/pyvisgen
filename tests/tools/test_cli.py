from pyvisgen.tools import cli


class TestCLI:
    def test_cli_help(self, runner) -> None:
        result = runner.invoke(cli.main, ["--help"])

        assert result.exit_code == 0

    def test_converter_cmd(self, tmp_path, mocker, runner):
        mock_callback = mocker.MagicMock()
        cli.main.commands["convert"].callback = mock_callback

        result = runner.invoke(cli.main, ["convert", str(tmp_path), "-t", "all"])

        assert result.exit_code == 0
        mock_callback.assert_called_once()

    def test_create_dataset_cmd(self, mocker, runner):
        mock_callback = mocker.MagicMock()
        cli.main.commands["simulate"].callback = mock_callback

        result = runner.invoke(cli.main, ["simulate", "tests/test_conf.toml"])

        assert result.exit_code == 0
        mock_callback.assert_called_once()

    def test_quickstart_cmd(self, tmp_path, mocker, runner):
        mock_callback = mocker.MagicMock()
        cli.main.commands["quickstart"].callback = mock_callback

        result = runner.invoke(cli.main, ["quickstart", str(tmp_path)])

        assert result.exit_code == 0
        mock_callback.assert_called_once()
