import torch
from numpy.testing import assert_array_equal, assert_raises

from pyvisgen.layouts.layouts import get_array_layout


class TestLayouts:
    def setup_class(self):
        from pathlib import Path

        import pandas as pd
        from astropy.coordinates import EarthLocation

        # Declare a reduced EHT test layout. This should suffice for
        # testing purposes
        self.eht_test_layout = pd.DataFrame(
            {
                "st_num": [0, 1, 2],
                "x": [2225037.1851, -1828796.2, -768713.9637],
                "y": [-5441199.162, -5054406.8, -5988541.7982],
                "z": [-2479303.4629, 3427865.2, 2063275.9472],
                "diam": [84.700000, 10.000000, 50.000000],
                "el_low": [15.000000, 15.000000, 15.000000],
                "el_high": [85.000000, 85.000000, 85.000000],
                "sefd": [110.0, 11900.0, 560.0],
                "altitude": [5030.0, 3185.0, 4640.0],
            }
        )

        # Load test layout from file
        self.test_layout = pd.read_csv(
            Path(__file__).parent.resolve() / "data/test_layout.txt", sep=r"\s+"
        )

        # Place test layout in Dortmund
        loc = EarthLocation.of_address("dortmund")
        self.test_layout["X"] += loc.value[0]
        self.test_layout["Y"] += loc.value[1]
        self.test_layout["Z"] += loc.value[2]

    def test_get_array_layout(self):
        layout = get_array_layout("eht")

        assert len(layout.st_num) == 8

        assert_array_equal(layout[:3].x, self.eht_test_layout.x)
        assert_array_equal(layout[:3].y, self.eht_test_layout.y)
        assert_array_equal(layout[:3].z, self.eht_test_layout.z)
        assert_array_equal(layout[:3].diam, self.eht_test_layout.diam)
        assert_array_equal(layout[:3].el_low, self.eht_test_layout.el_low)
        assert_array_equal(layout[:3].el_high, self.eht_test_layout.el_high)
        assert_array_equal(layout[:3].sefd, self.eht_test_layout.sefd)
        assert_array_equal(layout[:3].altitude, self.eht_test_layout.altitude)

        layout = get_array_layout("vlba")

        assert len(layout.st_num) == 10

        layout = get_array_layout("vla")

        assert len(layout.st_num) == 27
        assert layout[:3].st_num.shape == torch.Size([3])

    def test_get_array_layout_dataframe(self):
        layout = get_array_layout(self.test_layout)

        assert_array_equal(layout.x, self.test_layout.X)
        assert_array_equal(layout.y, self.test_layout.Y)
        assert_array_equal(layout.z, self.test_layout.Z)
        assert_array_equal(layout.diam, self.test_layout.dish_dia)
        assert_array_equal(layout.el_low, self.test_layout.el_low)
        assert_array_equal(layout.el_high, self.test_layout.el_high)
        assert_array_equal(layout.sefd, self.test_layout.SEFD)
        assert_array_equal(layout.altitude, self.test_layout.altitude)

    def test_get_array_layout_raise(self):
        # Converting the test DataFrame to a dict should raise
        # a TypeError, since only str, pathlib.Path or pd.DataFrames
        # are allowed
        assert_raises(TypeError, get_array_layout, self.test_layout.to_dict)

    def test_get_array_names(self):
        from pyvisgen.layouts.layouts import get_array_names

        test = ["vla", "vlba", "eht", "alma"]

        assert set(test).issubset(get_array_names())
