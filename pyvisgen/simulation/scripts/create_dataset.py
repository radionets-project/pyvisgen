import click
from pyvisgen.simulation.data_set import simulate_data_set
from pyvisgen.gridding.gridder import create_gridded_data_set


@click.command()
@click.argument("configuration_path", type=click.Path(exists=True, dir_okay=False))
@click.option('--job_id', required=False, type=int, default=None)
@click.option('--n', required=False, type=int, default=None)
@click.option(
    "--mode",
    type=click.Choice(
        [
            "simulate",
            "gridding",
            "slurm",
        ],
        case_sensitive=False,
    ),
    default="simulate",
)
def main(configuration_path, mode, job_id=None, n=None):
    if mode == "simulate":
        simulate_data_set(configuration_path)
    if mode == "slurm":
        simulate_data_set(configuration_path, slurm=True, job_id=job_id, n=n)
    if mode == "gridding":
        create_gridded_data_set(configuration_path)


if __name__ == "__main__":
    main()
