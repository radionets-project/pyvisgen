from importlib.metadata import entry_points
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict, Type

from pyvisgen import __version__


class _DiscoverPlugins:
    def _gridding(self) -> Dict[str, Type]:
        """Classmethod that discovers all available gridding plugins."""
        self.plugins = {}

        eps = entry_points()
        if hasattr(eps, "select"):
            gridding_plugins = eps.select(group="pyvisgen.gridding")

        for entry_point in gridding_plugins:
            try:
                plugin_class = entry_point.load()
                self.plugins[entry_point.name] = plugin_class
            except ImportError as e:
                print(f"Failed to load plugin {entry_point.name}: {e}")

        return self.plugins

    @classmethod
    def get_gridder(cls, name="gridder"):
        """Get a specific gridding plugin by name."""
        instance = cls()

        plugins = instance._gridding()
        if not list(plugins.keys()):
            raise ValueError(
                "No plugins available in 'pyvisgen.gridding' entry point group! "
                "Make sure you have installed a package providing plugins compatible "
                f"with pyvisgen {__version__}, e.g. pyvisgrid (uv pip install pyvisgrid)!"
            )
        if name not in plugins:
            raise ValueError(
                f"Plugin '{name}' not found. Available: {list(plugins.keys())}"
            )

        return plugins[name]
