import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PatchCollection, PathCollection

from pyvisgen.layouts import ArrayDisplay


class TestArrayDisplay:
    def test_set_unit(self, mocker, mock_stations):
        stations = mock_stations

        mocker.patch.object(ArrayDisplay, "_add_radial_grid")
        mocker.patch.object(ArrayDisplay, "_add_antennas")

        disp = ArrayDisplay(stations)

        attrs = [
            "x",
            "y",
            "z",
            "r",
            "axis_unit_label",
            "xlabel",
            "ylabel",
        ]

        for attr in attrs:
            assert hasattr(disp, attr)

    def test_set_unit_auto_deg(self, mocker, mock_stations):
        stations = mock_stations

        stations.x *= 1e6
        stations.y *= 1e6
        stations.z *= 1e6

        mocker.patch.object(ArrayDisplay, "_add_radial_grid")
        mocker.patch.object(ArrayDisplay, "_add_antennas")

        disp = ArrayDisplay(stations, unit="auto")

        assert disp.axis_unit_label == "deg"

    def test_set_unit_auto_km(self, mocker, mock_stations):
        stations = mock_stations

        stations.x *= 1e3
        stations.y *= 1e3
        stations.z *= 1e3

        mocker.patch.object(ArrayDisplay, "_add_radial_grid")
        mocker.patch.object(ArrayDisplay, "_add_antennas")

        disp = ArrayDisplay(stations, unit="auto")

        assert disp.axis_unit_label == "km"

    def test_set_unit_auto_m(self, mocker, mock_stations):
        stations = mock_stations

        mocker.patch.object(ArrayDisplay, "_add_radial_grid")
        mocker.patch.object(ArrayDisplay, "_add_antennas")

        disp = ArrayDisplay(stations, unit="auto")

        assert disp.axis_unit_label == "m"

    def test_set_unit_km(self, mocker, mock_stations):
        stations = mock_stations

        mocker.patch.object(ArrayDisplay, "_add_radial_grid")
        mocker.patch.object(ArrayDisplay, "_add_antennas")

        disp = ArrayDisplay(stations, unit="km")

        assert disp.unit == "km"
        assert disp.axis_unit_label == "km"

    def test_set_unit_deg(self, mocker, mock_stations):
        stations = mock_stations

        mocker.patch.object(ArrayDisplay, "_add_radial_grid")
        mocker.patch.object(ArrayDisplay, "_add_antennas")

        disp = ArrayDisplay(stations, unit="deg")

        assert disp.unit == "deg"
        assert disp.axis_unit_label == "deg"

    def test_set_unit_m(self, mocker, mock_stations):
        stations = mock_stations

        mocker.patch.object(ArrayDisplay, "_add_radial_grid")
        mocker.patch.object(ArrayDisplay, "_add_antennas")

        disp = ArrayDisplay(stations, unit="m")

        assert disp.unit == "m"
        assert disp.axis_unit_label == "m"

    def test_set_unit_raises(self, mocker, mock_stations):
        stations = mock_stations

        mocker.patch.object(ArrayDisplay, "_add_radial_grid")
        mocker.patch.object(ArrayDisplay, "_add_antennas")

        wrong_unit = "Jy"

        with pytest.raises(ValueError, match=f"unit {wrong_unit!r} is unknown"):
            ArrayDisplay(stations, unit="Jy")

    @pytest.mark.parametrize("marker_type", ["fixed", "diameter"])
    def test_add_antennas(self, marker_type, mocker, mock_stations):
        stations = mock_stations

        mocker.patch.object(ArrayDisplay, "_add_radial_grid")

        disp = ArrayDisplay(stations, marker_type=marker_type)

        assert disp.marker_type == marker_type
        assert hasattr(disp, "marker_sizes")
        assert disp.marker_sizes[0] == disp.marker_size
        assert hasattr(disp, "antennas")
        assert isinstance(disp.antennas, PathCollection)

    @pytest.mark.parametrize("marker_color", ["blue", None])
    def test_add_antennas_color(self, marker_color, mocker, mock_stations):
        import cmasher as cmr
        from cycler import cycler

        colors = cmr.take_cmap_colors("tab10", None, return_fmt="hex")
        plt.rc("axes", prop_cycle=cycler(color=colors))

        stations = mock_stations

        mocker.patch.object(ArrayDisplay, "_add_radial_grid")

        disp = ArrayDisplay(stations, marker_color=marker_color)

        # If None, color is selected from mpl prop_cycle
        if marker_color is None:
            marker_color = "#1F77B4"

        assert disp.marker_color == marker_color

    @pytest.mark.parametrize("diam", [np.nan, -1])
    def test_dish_marker_sizes_raises_non_finite(self, diam, mocker, mock_stations):
        stations = mock_stations
        stations.diam = np.full_like(stations.diam, diam)

        mocker.patch.object(ArrayDisplay, "_add_radial_grid")

        with pytest.raises(
            ValueError, match="station diameters must be finite and > 0"
        ):
            ArrayDisplay(stations, marker_type="diameter")

    def test_dish_marker_sizes_raises_scale(self, mocker, mock_stations):
        stations = mock_stations
        stations.diam = np.arange(len(stations.diam)) + 1

        mocker.patch.object(ArrayDisplay, "_add_radial_grid")

        with pytest.raises(
            ValueError, match="chosen diameter_scale creates non-positive marker sizes"
        ):
            ArrayDisplay(stations, marker_type="diameter", diameter_scale=-1)

    def test_add_radial_grid(self, mock_locations):
        loc = mock_locations

        class MockArrayDisplay:
            _add_radial_grid = ArrayDisplay._add_radial_grid

            def __init__(self):

                self.x = loc.x
                self.y = loc.y

                self.fig, self.axes = plt.subplots()

            def _grid_step(self, val):
                return 2.0

        disp = MockArrayDisplay()
        assert len(disp.axes.collections) == 0

        disp._add_radial_grid(center=(0, 0), color="red", alpha=0.75, linewidth=3.0)
        assert len(disp.axes.collections) == 1

        collection = disp.axes.collections[0]

        assert isinstance(collection, PatchCollection)
        assert collection.get_alpha() == 0.75
        assert collection.get_linewidth()[0] == 3.0  # ty:ignore[not-subscriptable]

        paths = collection.get_paths()
        assert len(paths) > 0

        plt.close(disp.fig)

    @pytest.mark.parametrize("step", [0.0, -1.0])
    def test_add_radial_grid_step_invalid(self, step, mock_locations):
        """Check that no grid is added when step is invalid"""
        loc = mock_locations

        class MockArrayDisplay:
            _add_radial_grid = ArrayDisplay._add_radial_grid

            def __init__(self):

                self.x = loc.x
                self.y = loc.y

                self.fig, self.axes = plt.subplots()

            def _grid_step(self, val):
                return -step  # negative value should be invalid

        disp = MockArrayDisplay()
        assert len(disp.axes.collections) == 0

        disp._add_radial_grid(center=(0, 0), color="red", alpha=0.75, linewidth=3.0)
        assert len(disp.axes.collections) == 0

        plt.close(disp.fig)
