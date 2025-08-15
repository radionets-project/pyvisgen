from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Group
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

if TYPE_CHECKING:
    from typing import Any, Dict, List, Optional


def create_progress_tracker(
    custom_configs: Optional[Dict[str, List[Any]]] = None,
) -> Dict[str, Any]:
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
            TextColumn("{task.description}"),
        ],
        "counting": [
            TextColumn("[bold green]Counting images: {task.percentage:.0f}%"),
            BarColumn(),
            TextColumn("({task.completed} of {task.total} bundles processed)"),
        ],
        "testing": [
            TextColumn(
                "[bold red]Pre-drawing and testing sample parameters: "
                "{task.percentage:.0f}%"
            ),
            BarColumn(),
            TextColumn(
                "({task.completed} of {task.total} [bold]valid[/] parameter sets created)"
            ),
        ],
        "bundles": [
            TextColumn("[bold blue]Progress for all Bundles: {task.percentage:.0f}%"),
            BarColumn(),
            TextColumn("({task.completed} of {task.total} bundles saved)"),
        ],
        "current_bundle": [
            TextColumn(
                "[bold purple]Progress for bundle {task.fields[name]}: {task.percentage:.0f}%"
            ),
            BarColumn(),
            TextColumn("({task.completed} of {task.total} steps done)"),
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
        Panel(Group(*_progress_bars.values())), progress_bars["overall"]
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
