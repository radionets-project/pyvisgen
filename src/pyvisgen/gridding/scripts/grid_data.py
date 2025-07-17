import click

from pyvisgen.gridding.utils import create_gridded_data_set


@click.command()
@click.option("-c", "--config", required=True, type=str, default=None)
def grid_data(config):
    create_gridded_data_set(config)


if __name__ == "__main__":
    grid_data()
