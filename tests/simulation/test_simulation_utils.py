from rich.console import Group
from rich.panel import Panel
from rich.progress import Progress

from pyvisgen.simulation.utils import create_progress_tracker


class TestProgresstracker:
    def test_default_configs(self):
        progress = create_progress_tracker()

        expected_keys = [
            "progress_bars",
            "group",
            "overall",
            "counting",
            "testing",
            "bundles",
            "current_bundle",
        ]

        assert set(expected_keys) == set(progress)

        pbars = progress["progress_bars"]
        assert isinstance(pbars, dict)

        for bar in list(expected_keys)[2:]:
            assert bar in pbars
            assert isinstance(pbars[bar], Progress)

        group = progress["group"]
        assert isinstance(group, Group)
        assert isinstance(group.renderables[0], Panel)
        assert pbars["overall"] in group.renderables

    def test_custom_config(self):
        custom_config = {
            "counting": [],  # empty
            "foo": [],
        }

        progress = create_progress_tracker(custom_config)

        assert "foo" in progress["progress_bars"]
        assert isinstance(progress["progress_bars"]["foo"], Progress)
        assert isinstance(progress["counting"], Progress)  # should still be Progress
