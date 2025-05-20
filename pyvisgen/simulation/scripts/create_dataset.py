import click

from pyvisgen.simulation.data_set import SimulateDataSet


@click.command()
@click.argument(
    "configuration_path",
    type=click.Path(exists=True, dir_okay=False),
)
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
@click.option(
    "-k",
    "--key",
    required=False,
    type=str,
    default="y",
    help="Key under which the images are saved in the HDF5 file",
)
@click.option("--slurm_job_id", required=False, type=int, default=None)
@click.option("--slurm_n", required=False, type=int, default=None)
@click.option(
    "--num_images",
    required=False,
    type=int,
    default=None,
    help="""
        Number of images in all bundles combined.
        Will skip the automatic count.
    """,
)
@click.option(
    "-p",
    "--multiprocess",
    required=False,
    help="""
        Number of processes to run in parallel while
        sampling and testing parameters. If -1 or
        'all', will use all available cores.
    """,
)
@click.option(
    "-s",
    "--stokes",
    required=False,
    type=str,
    default="I",
    help="""Stokes component to grid/simulate.""",
)
def main(
    configuration_path: str | click.Path,
    mode: str,
    key: str = "y",
    slurm_job_id=None,
    slurm_n=None,
    date_fmt="%d-%m-%Y %H:%M:%S",
    num_images: int | None = None,
    multiprocess: int | str = 1,
    stokes: str = "I",
):
    if mode == "simulate":
        SimulateDataSet.from_config(
            configuration_path,
            image_key=key,
            grid=False,
            date_fmt=date_fmt,
            num_images=num_images,
            multiprocess=multiprocess,
        )
    if mode == "slurm":
        SimulateDataSet.from_config(
            configuration_path,
            image_key=key,
            slurm=True,
            slurm_job_id=slurm_job_id,
            slurm_n=slurm_n,
            date_fmt=date_fmt,
            num_images=num_images,
        )
    if mode == "gridding":
        SimulateDataSet.from_config(
            configuration_path,
            image_key=key,
            grid=True,
            date_fmt=date_fmt,
            num_images=num_images,
            multiprocess=multiprocess,
        )


if __name__ == "__main__":
    main()
