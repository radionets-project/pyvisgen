import logging

from rich.logging import RichHandler


def setup_logger(**kwargs):
    """Basic logging setup. Uses :class:`~rich.logging.RichHandler`
    for formatting and highlighting of the log.

    Parameters
    ----------
    **kwargs
        Keyword arguments for :class:`~rich.logging.RichHandler`.

    Returns
    -------
    logging.Logger
        Logger object using :class:`~rich.logging.RichHandler`
        for formatting and highlighting.

    See Also
    --------
    :class:`~rich.logging.RichHandler` :
        Rich's builtin logging handler for more information on
        allowed keyword arguments.
    """
    FORMAT = "%(message)s"

    logging.basicConfig(
        level="NOTSET",
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, **kwargs)],
    )

    return logging.getLogger("rich")
