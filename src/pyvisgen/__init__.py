"""
pyvisgen - Python implementation of the VISGEN tool.

Licensed under a MIT style license - see LICENSE
"""

from rich.traceback import install

from .version import __version__

__all__ = ["__version__"]


install(show_locals=False)
