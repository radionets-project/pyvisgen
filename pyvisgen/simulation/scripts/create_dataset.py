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
@click.option("-j", "--job_id", required=False, type=int, default=None)
@click.option("--n", required=False, type=int, default=None)
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
def main(
    configuration_path: str | click.Path,
    mode: str,
    key: str = "y",
    job_id=None,
    n=None,
    date_fmt="%d-%m-%Y %H:%M:%S",
    num_images: int | None = None,
):
    if mode == "simulate":
        SimulateDataSet.from_config(
            configuration_path,
            image_key=key,
            grid=False,
            date_fmt=date_fmt,
            num_images=num_images,
        )
    if mode == "slurm":
        SimulateDataSet.from_config(
            configuration_path,
            image_key=key,
            slurm=True,
            job_id=job_id,
            n=n,
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
        )


if __name__ == "__main__":
    main()
