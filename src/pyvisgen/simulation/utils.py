from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Group
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

if TYPE_CHECKING:
    from typing import Any


def create_progress_tracker(
    custom_configs: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    """
    Factory function to create customizable progress trackers

    Args:
        custom_configs: Custom column configurations for progress bars
        theme_color: Color theme for progress bars

    Returns:
        Dictionary containing progress bars and group
    """

    # Default configurations
    configs = {
        "overall": [
            SpinnerColumn("dots"),
            TimeElapsedColumn(),
            TextColumn(
                "[#aaaaaa]{task.description} ({task.completed} of {task.total} tasks completed)"  # noqa: E501
            ),
        ],
        "counting": [
            TextColumn("[bold #40a02b]Counting images: {task.percentage:>29.0f}%"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn(" bundles processed •"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        ],
        "testing": [
            TextColumn(
                "[bold #e64553]Pre-drawing and testing sample parameters: "
                "{task.percentage:.0f}%"
            ),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[bold]valid[/] parameter sets created •"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        ],
        "bundles": [
            TextColumn(
                "[bold #04a5e5]Progress for all Bundles: {task.percentage:>20.0f}%"
            ),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("saved •"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        ],
        "current_bundle": [
            TextColumn(
                "[bold #7287fd]Progress for bundle {task.fields[name]}: {task.percentage:>23.0f}%"  # noqa: E501
            ),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("images processed •"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        ],
    }

    # Override with custom configs if provided
    if custom_configs:
        configs.update(custom_configs)

    # Create progress bars
    progress_bars = {name: Progress(*columns) for name, columns in configs.items()}
    _progress_bars = progress_bars.copy()
    _progress_bars.pop("overall")

    progress_group = Group(
        Panel(
            Group(*_progress_bars.values()), title="Task Progress", title_align="left"
        ),
        progress_bars["overall"],
    )

    return {
        "progress_bars": progress_bars,
        "group": progress_group,
        # Also return default progress bars again
        # separately for quick access
        "overall": progress_bars["overall"],
        "counting": progress_bars["counting"],
        "testing": progress_bars["testing"],
        "bundles": progress_bars["bundles"],
        "current_bundle": progress_bars["current_bundle"],
    }
