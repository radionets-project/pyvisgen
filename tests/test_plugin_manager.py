import logging

from numpy.testing import assert_raises

from pyvisgen._plugin_manager import Manager, PluginManager
from pyvisgen.utils.config import read_data_set_conf


def test_get_avail_plugins():
    manager = Manager()
    assert_raises(
        ValueError,
        manager._get_avail_plugins,
        "pyvisgen.this_group_does_not_exist",
    )

def test_get_gridder():
    manager = PluginManager()

    assert_raises(
        ValueError,
        manager.get_gridder,
        "this_gridder_plugin_does_not_exist"
    )

def test_gridder_fallback(caplog):
    """Tests fallback if gridder plugin is not available
    when called in pyvisgen.dataset.SimulateDataset
    """
    from pyvisgen.dataset import SimulateDataSet

    s = SimulateDataSet

    config = read_data_set_conf("tests/test_conf.toml")
    config["gridder"] = "this_gridder_plugin_does_not_exist"

    with caplog.at_level(logging.WARN):
        s.from_config(config)

    assert "Falling back to default gridder!" in caplog.text
