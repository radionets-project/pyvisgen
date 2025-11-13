import sysconfig
from pathlib import Path

import rich_click as click
import toml
from rich.pretty import pretty_repr

from pyvisgen import __version__
from pyvisgen.utils import setup_logger


@click.command()
@click.argument(
    "config_path",
    type=click.Path(dir_okay=True),
)
@click.option(
    "-y",
    "--yes",
    "overwrite",
    type=bool,
    is_flag=True,
    help="Overwrite file if it already exists.",
)
def quickstart(
    config_path: str | Path,
    overwrite: bool = False,
) -> None:
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
    log = setup_logger(namespace=__name__, tracebacks_suppress=[click])

    msg = f"This is the pyvisgen [blue]v{__version__}[/] quickstart tool"
    log.info(msg, extra={"markup": True, "highlighter": None})
    log.info((len(msg) - len("[blue][/]")) * "=")

    if isinstance(config_path, str):
        config_path = Path(config_path)

    root = sysconfig.get_path("data", sysconfig.get_default_scheme())
    default_config_path = Path(root + "/share/configs/default_data_set.toml")

    with open(default_config_path) as f:
        default_config = toml.load(f)

    log.info("Loading default pyvisgen configuration:")
    log.info(pretty_repr(default_config))

    if config_path.is_dir():
        config_path /= "pyvisgen_default_data_set_config.toml"

    # write_file is used below; the following if statement acts as
    # a switch, toggling write_file to False if the user does not
    # wish to overwrite
    write_file = True
    if config_path.is_file() and not overwrite:
        log.info("")
        write_file = click.confirm(
            f"{config_path} already exists! Overwrite?", default=False
        )

    if write_file:
        with open(config_path, "w") as f:
            toml.dump(default_config, f)

        log.info(
            f"Configuration file was successfully written to {config_path.absolute()}",
        )
    else:
        log.warning("No output file was written!")


if __name__ == "__main__":
    quickstart()
