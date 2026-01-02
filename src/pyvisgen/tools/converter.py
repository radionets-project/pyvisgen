import rich_click as click

from pyvisgen.io import DataConverter


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
    epilog="""
    See https://pyvisgen.readthedocs.io/en/latest/api/pyvisgen.io.dataconverter.DataConverter.html
    for more information on extra arguments.
    """,
)
@click.pass_context
@click.argument(
    "input_dir",
    type=click.Path(exists=True, dir_okay=True),
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(dir_okay=True),
    default=None,
    help="Output directory. Defaults to input directory.",
)
@click.option(
    "--input-format",
    type=click.Choice(["h5", "wds"], case_sensitive=False),
    default="h5",
    help="Data format of the input dataset.",
    show_default=True,
)
@click.option(
    "--output-format",
    type=click.Choice(["h5", "wds"], case_sensitive=False),
    default="wds",
    help="Data format of the output dataset.",
    show_default=True,
)
@click.option(
    "--dataset_type",
    "-d",
    type=click.Choice(
        [
            "all",
            "train",
            "valid",
            "test",
        ],
        case_sensitive=False,
    ),
    default="all",
    show_default=True,
)
def main(
    ctx: click.Context,
    input_dir: str,
    output_dir: str | None,
    input_format: str,
    output_format: str,
    dataset_type: str,
):
    """Data format conversion tool for pyvisgen."""
    input_format = input_format.lower()
    output_format = output_format.lower()
    dataset_type = dataset_type.lower()

    if input_format == output_format:
        raise ValueError(
            "Expected input and output format to be different but "
            f"got '{input_format}', '{output_format}'."
        )

    output_dir = output_dir or input_dir  # If output_dir is None, use input_dir

    kwargs = {}
    args = ctx.args
    if args:
        if len(args) % 2 != 0:
            raise click.UsageError(
                "Extra arguments must be key-value pairs: --key value", ctx
            )
        # Remove (left) dashes from flags, and replace dashes in args
        # with underscores so that, e.g. amp-phase will become amp_phase
        # (just in case)
        keys = [k.lstrip("-").replace("-", "_") for k in args[::2]]
        kwargs = dict(zip(keys, args[1::2]))

    # Get correct converter from DataConverter attribute
    converter = getattr(DataConverter, f"from_{input_format}")
    converter(input_dir, dataset_type=dataset_type).to(
        output_dir, output_format=output_format, **kwargs
    )


if __name__ == "__main__":
    main()
