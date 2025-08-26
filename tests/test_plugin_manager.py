from numpy.testing import assert_raises

from pyvisgen._plugin_manager import Manager, PluginManager


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
        "this_plugin_does_not_exist"
    )
