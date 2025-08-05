import logging

from rich.logging import RichHandler


def setup_logger(**kwargs):
    FORMAT = "%(message)s"

    logging.basicConfig(
        level="NOTSET",
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, **kwargs)],
    )

    return logging.getLogger("rich")
