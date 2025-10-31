from contextlib import contextmanager

try:
    from codecarbon import OfflineEmissionsTracker

    _CODECARBON_AVAIL = True
except ImportError:
    _CODECARBON_AVAIL = False

__all__ = ["carbontracker"]


@contextmanager
def carbontracker(config):
    """
    Context manager for tracking carbon emissions during code execution.

    This context manager creates a CodeCarbon OfflineEmissionsTracker
    based on the provided configuration. If CodeCarbon
    is available and enabled in the config, it will track
    carbon emissions for the code block within the context.
    If CodeCarbon ist not available or disabled, the tracking
    will be skipped.

    The tracker is mainly used in the dataset creation tool of pyvisgen.

    Parameters
    ----------
    config : :class:`~pyvisgen.io.Config`
        Pyvisgen's configuration BaseModel that contains the
        codecarbon BaseModel.

    Yields
    ------
    OfflineEmissionsTracker
        An active emissions tracker instance if CodeCarbon is
        available and enabled in the configuration.

    None
        If CodeCarbon is not available or disabled in the configuration.

    Example
    -------
    >>> with carbontracker(config) as tracker:
    ...     SimulateDataSet.from_config(config)

    The tracker automatically stops after the context block and
    saves the result to the path specified under the codecarbon.output_path
    key set in the config file.

    Note
    ----
    Requires CodeCarbon to work (this is checked at runtime).
    """
    if _CODECARBON_AVAIL and config.codecarbon:
        tracker = OfflineEmissionsTracker(**config.codecarbon.model_dump())
        try:
            yield tracker.start()
        finally:
            tracker.stop()
    else:
        yield None
