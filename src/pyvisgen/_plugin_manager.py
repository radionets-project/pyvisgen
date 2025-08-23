from __future__ import annotations

from abc import ABC
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

from pyvisgen import __version__
from pyvisgen.utils.logging import setup_logger

LOGGER = setup_logger(__name__)


class Manager(ABC):
    "Abstract base class for the pyvisgen plugin manager."

    def _get_avail_plugins(self, group: str) -> dict[str, Callable]:
        """Classmethod that discovers all available plugins
        in a given entry point group.

        Returns
        -------
        dict
            Dictionary containing all available plugins
            in the given entry point group.
        """
        self.plugins = {}

        eps = entry_points()
        if hasattr(eps, "select"):
            gridding_plugins = eps.select(group=group)

        for entry_point in gridding_plugins:
            try:
                plugin_class = entry_point.load()
                self.plugins[entry_point.name] = plugin_class
            except ImportError as e:
                LOGGER.warn(f"Failed to load plugin {entry_point.name} in {group}: {e}")

        return self.plugins

    def _get_plugin(self, name, group):
        plugins = self._get_avail_plugins(group=group)
        if not list(plugins.keys()):
            raise ValueError(
                "No plugins available in entry point group 'pyvisgen.gridding'! "
                "Make sure you have installed a package providing plugins compatible "
                f"with pyvisgen {__version__}, e.g. pyvisgrid.gridder from pyvisgrid "
                "(uv pip install pyvisgrid)!"
            )
        if name not in plugins:
            raise ValueError(
                f"Plugin '{name}' not found. Available: {list(plugins.keys())}"
            )

        return plugins[name]


class PluginManager(Manager):
    """Plugin manager class for pyvisgen."""

    @classmethod
    def get_gridder(cls, name: str = "pyvisgrid.gridder") -> Callable:
        """Get a specific gridding plugin by name.

        Parameters
        ----------
        name : str, optional
            Name of the gridder plugin. The plugin has to be part of
            the 'pyvisgen.gridding' entry point group. Raises a ValueError
            if no plugin is found in that group. Default: ``'pyvisgrid.gridder'``

        Return
        ------
        instance or callable
            Gridder plugin for the given name.

        Raises
        ------
        ValueError
            If no plugins are found in the 'pyvisgen.gridding' entry point group.
        ValueError
            If plugin 'name' was not found.

        Notes
        -----
        The ``'pyvisgrid.gridder`` plugin can be installed with pyvisgrid:

        .. code-block: shell-session

           $ uv pip install pyvisgrid

        """
        instance = cls()
        gridding_plugin = instance._get_plugin(name, group="pyvisgen.gridding")

        return gridding_plugin

    @classmethod
    def get_ft(cls, name: str) -> Callable:
        """Get a specific gridding plugin by name.

        Parameters
        ----------
        name : str, optional
            Name of the Fourier transform plugin. The plugin has to be part of
            the 'pyvisgen.ft' entry point group. Raises a ValueError
            if no plugin is found in that group.

        Return
        ------
        instance or callable
            Fourier transform plugin for the given name.

        Raises
        ------
        ValueError
            If no plugins are found in the 'pyvisgen.ft' entry point group.
        ValueError
            If plugin 'name' was not found.
        """
        instance = cls()
        ft_plugin = instance.__get_plugin(name, group="pyvisgen.ft")

        return ft_plugin
