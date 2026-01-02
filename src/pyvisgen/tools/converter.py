import rich_click as click

from pyvisgen.io import DataConverter


@click.command()
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
    help="Format of the input dataset.",
    show_default=True,
)
@click.option(
    "--output-format",
    type=click.Choice(["h5", "wds"], case_sensitive=False),
    default="wds",
    help="Format of the output dataset.",
    show_default=True,
)
@click.option(
    "--dataset_type",
    "-t",
    type=click.Choice(
        [
            "all",
            "train",
            "valid",
            "test",
        ],
        case_sensitive=False,
    ),
    multiple=True,
    default="all",
    help="""Choose between different splits of the datasets to convert.
        'all' converts 'train', 'valid', and 'test' splits in one go.
        """,
    show_default=True,
)
@click.option(
    "--amp-phase",
    is_flag=True,
    help="""
        Whether data contains amplitude/phase or real/imaginary data.
        This is important for metadata in formats like [bold blue]WebDataset[/] or
        [bold blue]PyTorch[/] pickle files.
    """,
)
@click.option(
    "--shard-pattern",
    type=str,
    default="%06d.tar",
    help="""
        Shard pattern for [bold blue]WebDataset[/] files. Must be a C-style format
        specifier with the '.tar' file suffix. Not applied for other
        data formats.
    """,
    show_default=True,
)
@click.option(
    "--compress",
    is_flag=True,
    help="""
        Whether to compress [bold blue]WebDataset[/] shards using gzip.
        Not applied for other data formats.
    """,
)
def main(
    input_dir: str,
    output_dir: str | None,
    input_format: str,
    output_format: str,
    dataset_type: str,
    amp_phase: bool,
    shard_pattern: str,
    compress: bool,
) -> None:
    """Data format conversion tool for pyvisgen."""
    input_format = input_format.lower()
    output_format = output_format.lower()
    dataset_type = [t.lower() for t in dataset_type]

    if input_format == output_format:
        raise click.BadParameter(
            "Expected input and output format to be different but "
            f"both are '{input_format}'."
        )

    output_dir = output_dir or input_dir  # If output_dir is None, use input_dir

    # Get correct converter from DataConverter attribute
    converter: DataConverter = getattr(DataConverter, f"from_{input_format}")
    converter(input_dir, dataset_type=dataset_type).to(
        output_dir,
        output_format=output_format,
        amp_phase=amp_phase,
        shard_pattern=shard_pattern,
        compress=compress,
    )


if __name__ == "__main__":
    main()
