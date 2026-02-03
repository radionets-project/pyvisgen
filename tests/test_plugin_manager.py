from importlib.metadata import EntryPoint, EntryPoints
from logging import WARNING

import pytest
from numpy.testing import assert_raises

from pyvisgen._plugin_manager import Manager, PluginManager


@pytest.fixture(scope="module")
def manager() -> Manager:
    return Manager()


@pytest.fixture(scope="module")
def plugin_manager() -> PluginManager:
    return PluginManager()


class TestManager:
    @pytest.fixture
    def group(self) -> str:
        return "pyvisgen.gridding"

    def test_get_avail_plugins_group_invalid(self, manager: Manager) -> None:
        group = "pyvisgen.this_group_does_not_exist"

        with pytest.raises(ValueError) as excinfo:
            manager._get_avail_plugins(group)

        assert f"Entry point group '{group}' not found!" in str(excinfo.value)

    def test_get_avail_plugins_entry_point_load(
        self, mocker, caplog, group: str, manager: Manager
    ) -> None:
        mock_ep_select = mocker.patch(
            "pyvisgen._plugin_manager.entry_points",
            return_value=EntryPoints(
                [EntryPoint(name="this_ep_does_not_exist", value="none", group=group)]
            ),
        )

        with caplog.at_level(WARNING):
            manager._get_avail_plugins(group)

        assert "Failed to load plugin" in caplog.text
        assert mock_ep_select.called

    def test_get_plugins_no_plugins(self, mocker, group: str, manager: Manager) -> None:
        mock_get_avail_plugins = mocker.patch.object(
            manager, "_get_avail_plugins", return_value={}
        )

        with pytest.raises(ValueError) as excinfo:
            manager._get_plugin("", group)

        mock_get_avail_plugins.assert_called_with(group=group)
        assert f"No plugins available in entry point group '{group}'!" in str(
            excinfo.value
        )


class TestPluginManager:
    def test_get_gridder(self, plugin_manager: PluginManager) -> None:
        gridder = plugin_manager.get_gridder("pyvisgrid.gridder")
        assert gridder.__name__ == "Gridder"  # type: ignore

    def test_get_gridder_raises(self, plugin_manager: PluginManager) -> None:
        assert_raises(
            ValueError, plugin_manager.get_gridder, "this_gridder_plugin_does_not_exist"
        )

    def test_get_ft(self, mocker, plugin_manager: PluginManager) -> None:
        mock_ft_plugin = mocker.MagicMock()
        mock_ft_plugin.__name__ = "CupyFinufft"
        mock_get_avail_plugins = mocker.patch(
            "pyvisgen._plugin_manager.Manager._get_avail_plugins",
            return_value={"radioft": mock_ft_plugin},
        )

        ft = plugin_manager.get_ft("radioft")

        mock_get_avail_plugins.assert_called_with(group="pyvisgen.ft")

        assert ft.__name__ == mock_ft_plugin.__name__  # type: ignore

    def test_get_ft_raises(self, plugin_manager: PluginManager) -> None:
        assert_raises(
            ValueError, plugin_manager.get_ft, "this_ft_plugin_does_not_exist"
        )
