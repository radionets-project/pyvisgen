"""
pyvisgen - Python implementation of the VISGEN tool.

Licensed under a MIT style license - see LICENSE
"""

import rich_click as click
from rich.traceback import install

from .version import __version__

__all__ = ["__version__"]


install(show_locals=False)


click.rich_click.MAX_WIDTH = 100
click.rich_click.STYLE_COMMANDS_TABLE_COLUMN_WIDTH_RATIO = (1, 7)
click.rich_click.TEXT_MARKUP = "rich"
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.STYLE_ERRORS_SUGGESTION = "magenta italic"
click.rich_click.ERRORS_SUGGESTION = (
    "Try running the '--help' flag for more information."
)
click.rich_click.ERRORS_EPILOGUE = "To find out more, visit [link=https://pyvisgen.readthedocs.io]https://pyvisgen.readthedocs.io[/link]"
