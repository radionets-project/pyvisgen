import toml
import sysconfig
from pathlib import Path

import click

try:
    from rich import print
except ImportError:
    pass

from pyvisgen import __version__


@click.command()
@click.argument(
    "config_path",
    type=click.Path(dir_okay=True),
)
def quickstart(config_path: str | Path) -> None:
    """Quickstart CLI tool for pyvisgen. Creates
    a copy of the default simulation configuration
    file at the specified path.

    Parameters
    ----------
    config_path : str or Path
        Path to write the config to.

    Notes
    -----
    If a directory is given, this tool will create
    a file called 'pyvisgen_default_data_set_config.toml'
    inside that directory.
    """
    msg = f"This is the pyvisgen v{__version__} quickstart tool"
    print(msg)
    print(len(msg) * "=", "\n")

    if isinstance(config_path, str):
        config_path = Path(config_path)

    root = sysconfig.get_path("data", sysconfig.get_default_scheme())
    default_config_path = Path(root + "/share/configs/default_data_set.toml")

    with open(default_config_path, "r") as f:
        default_config = toml.load(f)

    print("Loading default pyvisgen configuration:")
    print(default_config)

    if config_path.is_dir():
        config_path /= "pyvisgen_default_data_set_config.toml"

    with open(config_path, "w") as f:
        toml.dump(default_config, f)

    print(
        "Configuration file was successfully written to",
        f"{config_path.absolute()}",
    )


if __name__ == "__main__":
    quickstart()
