import click
from pyvisgen.simulation.data_set import simulate_data_set
from pyvisgen.gridding.gridder import create_gridded_data_set


@click.command()
@click.argument("configuration_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--mode",
    type=click.Choice(
        [
            "simulate",
            "gridding",
        ],
        case_sensitive=False,
    ),
    default="simulate",
)
def main(configuration_path, mode):
    if mode == "simulate":
        print("hi")
        simulate_data_set(configuration_path)
    if mode == "gridding":
        create_gridded_data_set(configuration_path)


if __name__ == "__main__":
    main()
