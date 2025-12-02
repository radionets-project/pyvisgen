import rich_click as click

from pyvisgen import __version__

from .create_dataset import main as create_dataset
from .quickstart import main as quickstart

click.rich_click.COMMAND_GROUPS = {
    "pyvisgen": [
        {
            "name": "Simulation",
            "commands": ["simulate"],
        },
        {
            "name": "Setup",
            "commands": ["quickstart"],
        },
    ]
}


@click.group(
    help=f"""
    This is the [spring_green3]pyvisgen[/]
    [cornflower_blue]v{__version__}[/] main CLI tool.
    """
)
def main():
    pass


main.add_command(quickstart, name="quickstart")
main.add_command(create_dataset, name="simulate")

if __name__ == "__main__":
    main()
