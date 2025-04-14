from pathlib import Path

import torch
from numpy.testing import assert_array_equal, assert_raises

from pyvisgen.utils.config import read_data_set_conf

torch.manual_seed(1)

CONFIG = "tests/test_conf.toml"

conf = read_data_set_conf(CONFIG)
out_path = Path(conf["out_path_fits"])
out_path.mkdir(parents=True, exist_ok=True)


def test_get_data():
    from pyvisgen.utils.data import load_bundles

    data = load_bundles(conf["in_path"])
    assert len(data) > 0


class TestSimulateDataSet:
    """Unit test class for :class:``pyvisgen.simulation.SimulateDataSet``."""

    def setup_class(self):
        """Set up common objects and variables for the following tests."""
        from pyvisgen.simulation.data_set import SimulateDataSet

        self.s = SimulateDataSet

    def test_run_no_slurm(self):
        self.s.from_config(CONFIG)

    def test_run_no_slurm_multiprocess(self):
        self.s.from_config(CONFIG, multiprocess="all")

    def test_run_no_slurm_num_images(self):
        self.s.from_config(CONFIG, num_images=50)

    def test_run_no_slurm_amp_phase_false(self):
        config = conf.copy()
        config["amp_phase"] = False
        self.s.from_config(config)

    def test_raise_valerr_conf_path(self):
        assert_raises(ValueError, self.s.from_config, 42)

    def test_run_dense(self):
        config = conf.copy()
        config["mode"] = "dense"
        assert_raises(ValueError, self.s.from_config, config)

    def test_run_polarization(self):
        config = conf.copy()
        config["polarization"] = "linear"
        self.s.from_config(config)

    def test_run_no_gridding(self):
        self.s.from_config(CONFIG, grid=False)


class TestVisLoop:
    """Unit test class for :func:``pyvisgen.simulation.vis_loop``."""

    def setup_class(self):
        """Set up common objects and variables for the following tests."""
        from pyvisgen.simulation.data_set import SimulateDataSet

        self.s = SimulateDataSet

    def test_vis_loop(self):
        import pyvisgen.fits.writer as writer
        from pyvisgen.simulation.visibility import vis_loop
        from pyvisgen.utils.data import load_bundles, open_bundles

        bundles = load_bundles(conf["in_path"])
        _, obs = self.s._get_obs_test(CONFIG)

        data = open_bundles(bundles[0])
        SI = torch.tensor(data[0])[None]
        vis_data = vis_loop(obs, SI, noisy=conf["noisy"], mode=conf["mode"])

        assert (vis_data[0].V_11[0]).dtype == torch.complex128
        assert (vis_data[0].V_22[0]).dtype == torch.complex128
        assert (vis_data[0].V_12[0]).dtype == torch.complex128
        assert (vis_data[0].V_21[0]).dtype == torch.complex128
        assert (vis_data[0].num).dtype == torch.float64
        assert (vis_data[0].base_num).dtype == torch.float64
        assert torch.is_tensor(vis_data[0].u)
        assert torch.is_tensor(vis_data[0].v)
        assert torch.is_tensor(vis_data[0].w)
        assert (vis_data[0].date).dtype == torch.float64

        # test num vis for time step 0
        # num_vis_theory = num_active_telescopes * (num_active_telescopes - 1) / 2
        # num_vis_calc = vis_data.base_num[vis_data.date == vis_data.date[0]].shape[0]
        # dunno what's going on here
        # assert num_vis_theory == num_vis_calc
        #

        out_path = Path(conf["out_path_fits"])
        out = out_path / Path("vis_0.fits")
        hdu_list = writer.create_hdu_list(vis_data, obs)
        hdu_list.writeto(out, overwrite=True)

    def test_vis_loop_grid(self):
        import pyvisgen.fits.writer as writer
        from pyvisgen.simulation.visibility import vis_loop
        from pyvisgen.utils.data import load_bundles, open_bundles

        bundles = load_bundles(conf["in_path"])
        _, obs = self.s._get_obs_test(CONFIG)

        data = open_bundles(bundles[0])
        SI = torch.tensor(data[0])[None]
        vis_data = vis_loop(obs, SI, noisy=conf["noisy"], mode="grid")

        assert (vis_data[0].V_11[0]).dtype == torch.complex128
        assert (vis_data[0].V_22[0]).dtype == torch.complex128
        assert (vis_data[0].V_12[0]).dtype == torch.complex128
        assert (vis_data[0].V_21[0]).dtype == torch.complex128
        assert (vis_data[0].num).dtype == torch.float64
        assert (vis_data[0].base_num).dtype == torch.float64
        assert torch.is_tensor(vis_data[0].u)
        assert torch.is_tensor(vis_data[0].v)
        assert torch.is_tensor(vis_data[0].w)
        assert (vis_data[0].date).dtype == torch.float64

        # test num vis for time step 0
        # num_vis_theory = num_active_telescopes * (num_active_telescopes - 1) / 2
        # num_vis_calc = vis_data.base_num[vis_data.date == vis_data.date[0]].shape[0]
        # dunno what's going on here
        # assert num_vis_theory == num_vis_calc
        #

        out_path = Path(conf["out_path_fits"])
        out = out_path / Path("vis_0.fits")
        hdu_list = writer.create_hdu_list(vis_data, obs)
        hdu_list.writeto(out, overwrite=True)

    def test_vis_loop_batch_size_auto(self):
        from pyvisgen.simulation.visibility import vis_loop
        from pyvisgen.utils.data import load_bundles, open_bundles

        bundles = load_bundles(conf["in_path"])
        _, obs = self.s._get_obs_test(CONFIG)

        data = open_bundles(bundles[0])
        SI = torch.tensor(data[0])[None]

        vis_data = vis_loop(
            obs,
            SI,
            noisy=conf["noisy"],
            mode=conf["mode"],
            batch_size="auto",
        )

        assert (vis_data[0].V_11[0]).dtype == torch.complex128
        assert (vis_data[0].V_22[0]).dtype == torch.complex128
        assert (vis_data[0].V_12[0]).dtype == torch.complex128
        assert (vis_data[0].V_21[0]).dtype == torch.complex128
        assert (vis_data[0].num).dtype == torch.float64
        assert (vis_data[0].base_num).dtype == torch.float64
        assert torch.is_tensor(vis_data[0].u)
        assert torch.is_tensor(vis_data[0].v)
        assert torch.is_tensor(vis_data[0].w)
        assert (vis_data[0].date).dtype == torch.float64

    def test_vis_loop_batch_size_invalid(self):
        from pyvisgen.simulation.visibility import vis_loop
        from pyvisgen.utils.data import load_bundles, open_bundles

        bundles = load_bundles(conf["in_path"])
        _, obs = self.s._get_obs_test(CONFIG)

        data = open_bundles(bundles[0])
        SI = torch.tensor(data[0])[None]

        assert_raises(
            ValueError,
            vis_loop,
            obs,
            SI,
            noisy=conf["noisy"],
            mode=conf["mode"],
            batch_size="abc",
        )

        assert_raises(
            ValueError,
            vis_loop,
            obs,
            SI,
            noisy=conf["noisy"],
            mode=conf["mode"],
            batch_size=20.0,
        )


class TestPolarization:
    """Unit test class for ``pyvisgen.simulation.visibility.Polarization``."""

    def setup_class(self):
        """Set up common objects and variables for the following tests."""
        from pyvisgen.simulation.data_set import SimulateDataSet
        from pyvisgen.simulation.observation import (
            DEFAULT_FIELD_KWARGS,
            DEFAULT_POL_KWARGS,
        )
        from pyvisgen.simulation.visibility import Polarization

        _, self.obs = SimulateDataSet._get_obs_test(conf)

        # set to default kwargs for tests, otherwise
        # some parameters of the config would be None
        self.obs.field_kwargs = DEFAULT_FIELD_KWARGS
        self.obs.pol_kwargs = DEFAULT_POL_KWARGS

        self.SI = torch.zeros((100, 100))
        self.SI[25::25, 25::25] = 1
        self.SI = self.SI[None, ...]

        self.si_shape = self.SI.shape
        self.im_shape = self.si_shape[1], self.si_shape[2]

        self.obs.img_size = self.im_shape[0]

        self.pol = Polarization(
            self.SI,
            sensitivity_cut=self.obs.sensitivity_cut,
            polarization=self.obs.polarization,
            device=self.obs.device,
            field_kwargs=self.obs.field_kwargs,
            **self.obs.pol_kwargs,
        )

    def test_polarization_circular(self):
        """Test circular polarization."""

        self.pol.__init__(
            self.SI,
            sensitivity_cut=self.obs.sensitivity_cut,
            polarization="circular",
            device=self.obs.device,
            field_kwargs=self.obs.field_kwargs,
            **self.obs.pol_kwargs,
        )

        assert self.pol.delta == 0
        assert self.pol.ax2.sum() == self.SI.sum() * 0.5
        assert self.pol.ay2.sum() == self.SI.sum() * 0.5

        B, mask, lin_dop, circ_dop = self.pol.stokes_matrix()

        assert mask.sum() == 9
        assert B.shape == torch.Size([9, 2, 2])
        assert mask.shape == self.im_shape
        assert lin_dop.shape == self.im_shape
        assert lin_dop.shape == self.im_shape

    def test_polarization_linear(self):
        """Test linear polarization."""

        self.pol.__init__(
            self.SI,
            sensitivity_cut=self.obs.sensitivity_cut,
            polarization="linear",
            device=self.obs.device,
            field_kwargs=self.obs.field_kwargs,
            **self.obs.pol_kwargs,
        )

        assert self.pol.delta == 0
        assert self.pol.ax2.sum() == self.SI.sum() * 0.5
        assert self.pol.ay2.sum() == self.SI.sum() * 0.5

        B, mask, lin_dop, circ_dop = self.pol.stokes_matrix()

        assert mask.sum() == 9
        assert B.shape == torch.Size([9, 2, 2])
        assert mask.shape == self.im_shape
        assert lin_dop.shape == self.im_shape
        assert lin_dop.shape == self.im_shape

    def test_polarization_amplitude(self):
        """Test random amplitude."""
        pol_kwargs = {"delta": 0, "amp_ratio": None, "random_state": 42}

        self.pol.__init__(
            self.SI,
            sensitivity_cut=self.obs.sensitivity_cut,
            polarization="linear",
            device=self.obs.device,
            field_kwargs=self.obs.field_kwargs,
            **pol_kwargs,
        )

        assert self.pol.ax2.sum() <= 9
        assert self.pol.ay2.sum() <= 9

    def test_polarization_field(self):
        """Test polarization.rand_polarization_field method."""
        pf = self.pol.rand_polarization_field(shape=self.im_shape)

        assert pf.shape == torch.Size([100, 100])

    def test_polarization_field_random_state(self):
        """Test polarization field method for a given random_state"""
        random_state = 42

        pf = self.pol.rand_polarization_field(
            shape=self.im_shape,
            random_state=random_state,
        )

        assert torch.random.initial_seed() == random_state
        assert pf.shape == torch.Size([100, 100])

    def test_polarization_field_shape(self):
        """Test polarization field method for type(shape) = int."""
        pf_ref = self.pol.rand_polarization_field(
            shape=self.im_shape,
            random_state=42,
        )

        pf = self.pol.rand_polarization_field(
            shape=self.im_shape[0],
            random_state=42,
        )

        assert pf.shape == torch.Size([100, 100])
        assert_array_equal(pf, pf_ref, strict=True)

        # assert len(shape) > 2 raises ValueError
        assert_raises(
            ValueError,
            self.pol.rand_polarization_field,
            shape=[100, 100, 100],
            random_state=42,
        )

    def test_polarization_field_order(self):
        """Test polarization field method for different orders."""

        pf_ref = self.pol.rand_polarization_field(
            shape=self.im_shape,
            random_state=42,
        )

        pf = self.pol.rand_polarization_field(
            shape=self.im_shape,
            random_state=42,
            order=[1, 1],
        )

        assert pf.shape == torch.Size([100, 100])
        # assert order = 1 and order = [1, 1] yield same images
        assert_array_equal(pf, pf_ref, strict=True)

        pf = self.pol.rand_polarization_field(
            shape=self.im_shape,
            random_state=42,
            order=(1, 1),
        )
        # assert order = (1, 1) and order = [1, 1] yield same images
        assert_array_equal(pf, pf_ref, strict=True)

        pf = self.pol.rand_polarization_field(
            shape=self.im_shape,
            random_state=42,
            order=[1],
        )
        # assert order = [1] and order = [1, 1] yield same images
        assert_array_equal(pf, pf_ref, strict=True)

        # assert different order creates different image
        pf = self.pol.rand_polarization_field(
            shape=self.im_shape, random_state=42, order=[10, 10]
        )
        # expected to raise an AssertionError
        assert_raises(AssertionError, assert_array_equal, pf, pf_ref)

        # assert len(order) > 2 raises ValueError
        assert_raises(
            ValueError,
            self.pol.rand_polarization_field,
            shape=self.im_shape,
            random_state=42,
            order=[10, 10, 10],
        )

    def test_polarization_field_scale(self):
        """Test polarization field method for different scales."""

        pf_ref = self.pol.rand_polarization_field(
            shape=self.im_shape,
            random_state=42,
        )

        # scale = None
        pf = self.pol.rand_polarization_field(
            shape=self.im_shape,
            random_state=42,
            scale=None,
        )

        # expected to raise an AssertionError
        assert_raises(AssertionError, assert_array_equal, pf, pf_ref)

        # scale = [0.25, 0.25]
        pf = self.pol.rand_polarization_field(
            shape=self.im_shape, random_state=42, scale=[0.25, 0.25]
        )

        # expected to raise an AssertionError
        assert_raises(AssertionError, assert_array_equal, pf, pf_ref)

    def test_polarization_field_threshold(self):
        """Test polarization field method for different threshold."""

        pf_ref = self.pol.rand_polarization_field(
            shape=self.im_shape,
            random_state=42,
        )

        pf = self.pol.rand_polarization_field(
            shape=self.im_shape,
            random_state=42,
            threshold=0.5,
        )

        # expected to raise an AssertionError
        assert_raises(AssertionError, assert_array_equal, pf, pf_ref)
