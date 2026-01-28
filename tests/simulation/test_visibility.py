from dataclasses import fields

import numpy as np
import pytest
import torch

from pyvisgen.simulation.visibility import (
    Polarization,
    Visibilities,
    vis_loop,
)


class TestVisibilities:
    def test_get_item(self, visibilities: Visibilities) -> None:
        idx = 0

        visibilities_item = visibilities[idx]

        assert isinstance(visibilities_item, Visibilities)

        for f in fields(Visibilities):
            np.testing.assert_array_equal(
                getattr(visibilities_item, f.name).detach().cpu(),
                getattr(visibilities, f.name)[idx].detach().cpu(),
            )

    def test_get_slice(self, visibilities: Visibilities) -> None:
        visibilities_items = visibilities[2:-2]

        assert isinstance(visibilities_items, Visibilities)
        assert visibilities_items.u.shape == torch.Size([6])

    def test_add(
        self, visibilities: Visibilities, visibilities_data: dict[str, torch.Tensor]
    ) -> None:
        orig_len = visibilities.u.shape[0]
        new_visibilities = Visibilities(**visibilities_data)

        visibilities.add(new_visibilities)

        assert (
            visibilities.u.shape[0] == orig_len + new_visibilities.u.shape[0]
        )  # torch.Size([20])

    def test_add_baseline_to_empty(
        self, visibilities: Visibilities, device: str
    ) -> None:
        dev = torch.device(device)

        empty = Visibilities(
            V_11=torch.tensor([], device=dev),
            V_22=torch.tensor([], device=dev),
            V_12=torch.tensor([], device=dev),
            V_21=torch.tensor([], device=dev),
            num=torch.tensor([], device=dev),
            base_num=torch.tensor([], device=dev),
            u=torch.tensor([], device=dev),
            v=torch.tensor([], device=dev),
            w=torch.tensor([], device=dev),
            date=torch.tensor([], device=dev),
            linear_dop=torch.tensor([], device=dev),
            circular_dop=torch.tensor([], device=dev),
        )

        empty.add(visibilities)

        assert empty.u.shape == visibilities.u.shape

    def test_get_values(self, visibilities: Visibilities) -> None:
        vis_data = visibilities.get_values()

        assert vis_data.shape == visibilities.V_11.shape + torch.Size([4])
        np.testing.assert_array_equal(
            vis_data[..., 0].detach().cpu(),
            visibilities.V_11.detach().cpu(),
        )
        np.testing.assert_array_equal(
            vis_data[..., 1].detach().cpu(),
            visibilities.V_22.detach().cpu(),
        )
        np.testing.assert_array_equal(
            vis_data[..., 2].detach().cpu(),
            visibilities.V_12.detach().cpu(),
        )
        np.testing.assert_array_equal(
            vis_data[..., 3].detach().cpu(),
            visibilities.V_21.detach().cpu(),
        )


class TestPolarization:
    def test_init(self, polarization: Polarization, polarization_data: dict) -> None:
        _, H, W = polarization_data["SI"].shape

        assert polarization.sensitivity_cut == polarization_data["sensitivity_cut"]
        assert polarization.device == polarization_data["device"]

        assert hasattr(polarization, "ax2")
        assert hasattr(polarization, "ay2")
        assert polarization.ax2.shape == torch.Size([H, W])
        np.testing.assert_array_equal(
            polarization.ax2.detach().cpu(),
            polarization_data["SI"][0].detach().cpu(),
        )
        assert polarization.ay2.all() == 0

        assert hasattr(polarization, "I")
        assert isinstance(polarization.I, torch.Tensor)
        assert polarization.I.shape == torch.Size([H, W, 4])
        np.testing.assert_array_equal(
            polarization.I.detach().cpu(),
            torch.zeros((H, W, 4)).detach().cpu(),
        )

    def test_init_random_state(self, mocker, polarization_data: dict) -> None:
        polarization_data["random_state"] = 42

        mock_torch_manual_seed = mocker.patch(
            "pyvisgen.simulation.visibility.torch.manual_seed",
            return_value=torch.manual_seed(42),
        )

        Polarization(**polarization_data)

        mock_torch_manual_seed.assert_called_with(42)

    @pytest.mark.parametrize(
        "pol_type,amp_ratio,delta",
        [
            ("linear", 0.5, 45.0),  # amp_ratio and delta floats
            ("linear", 0.5, 90),  # delta int
            (
                "linear",
                torch.rand(1),
                torch.randint(low=0, high=90, size=(1,)),
            ),  # random values for amp_ratio and delta
            ("linear", -1, 90),  # amp_ratio negative, should draw random value
            ("circular", 0.5, 45.0),  # circular cases
            ("circular", 0.5, 90),
            (
                "circular",
                torch.rand(1),
                torch.randint(low=0, high=90, size=(1,)),
            ),
            ("circular", -1, 90),
        ],
    )
    def test_init_polarization(
        self,
        pol_type: str,
        amp_ratio: int | float,
        delta: int | float,
        mocker,
        polarization_data: dict,
    ) -> None:
        polarization_data["polarization"] = pol_type
        polarization_data["amp_ratio"] = amp_ratio
        polarization_data["delta"] = delta

        _, H, W = polarization_data["SI"].shape

        mock_rand_polarization_field = mocker.patch.object(
            Polarization,
            "rand_polarization_field",
            reurn_value=torch.rand((H, W)),
        )

        mock_torch_rand = mocker.patch(
            "pyvisgen.simulation.visibility.torch.rand",
            return_value=torch.rand(1).to(polarization_data["device"]),
        )

        pol = Polarization(**polarization_data)

        mock_rand_polarization_field.assert_called_with(
            [H, W], **polarization_data["field_kwargs"]
        )

        assert hasattr(pol, "ax2")
        assert hasattr(pol, "ay2")
        assert pol.ax2.shape == torch.Size([H, W])

        np.testing.assert_allclose(
            (pol.ax2 + pol.ay2).detach().cpu(),
            polarization_data["SI"][0].detach().cpu(),
        )

        if amp_ratio < 0:
            mock_torch_rand.assert_called_with(1)

    def test_linear(self, mocker, polarization_data: dict) -> None:
        mocker.patch.object(
            Polarization,
            "rand_polarization_field",
            reurn_value=None,
        )

        polarization_data["polarization"] = "linear"
        pol = Polarization(**polarization_data)

        assert pol.I.all() == 0.0

        mock_deg2rad = mocker.patch(
            "pyvisgen.simulation.visibility.torch.deg2rad",
            return_value=torch.deg2rad(pol.delta),
        )
        pol.linear()

        np.testing.assert_allclose(
            pol.I[..., 0].detach().cpu(),
            (pol.ax2 + pol.ay2).detach().cpu(),
        )
        mock_deg2rad.assert_called_with(pol.delta)

    def test_circular(self, mocker, polarization_data: dict) -> None:
        mocker.patch.object(
            Polarization,
            "rand_polarization_field",
            reurn_value=None,
        )

        polarization_data["polarization"] = "circular"
        pol = Polarization(**polarization_data)

        assert pol.I.all() == 0.0

        mock_deg2rad = mocker.patch(
            "pyvisgen.simulation.visibility.torch.deg2rad",
            return_value=torch.deg2rad(pol.delta),
        )
        pol.circular()

        np.testing.assert_allclose(
            pol.I[..., 0].detach().cpu(),
            (pol.ax2 + pol.ay2).detach().cpu(),
        )
        mock_deg2rad.assert_called_with(pol.delta)

    def test_rand_polarization_field(
        self, polarization: Polarization, field_kwargs: dict
    ) -> None:
        H, W, _ = polarization.SI.shape
        im = polarization.rand_polarization_field([H, W], **field_kwargs)

        assert im.shape == torch.Size([H, W])

    def test_rand_polarization_field_random_state(
        self, mocker, polarization, field_kwargs: dict
    ) -> None:
        H, W, _ = polarization.SI.shape

        field_kwargs["random_state"] = 42

        mock_torch_manual_seed = mocker.patch(
            "pyvisgen.simulation.visibility.torch.manual_seed",
            return_value=torch.manual_seed(42),
        )
        polarization.rand_polarization_field([H, W], **field_kwargs)

        mock_torch_manual_seed.assert_called_with(42)

    @pytest.mark.parametrize("shape", [16, 32, [16, 16], [16, 32], (16), (32, 32)])
    def test_rand_polarization_field_shape_instance(
        self, shape: int | list, polarization: Polarization, field_kwargs: dict
    ) -> None:
        im = polarization.rand_polarization_field(shape, **field_kwargs)

        if isinstance(shape, int):
            shape = [shape, shape]

        assert im.shape == torch.Size(shape)

    def test_rand_polarization_field_shape_wrong_dim(
        self, polarization: Polarization, field_kwargs: dict
    ) -> None:
        with pytest.raises(ValueError) as excinfo:
            polarization.rand_polarization_field([1, 16, 16], **field_kwargs)

        assert "Expected len of 'shape' to be 2!" in str(excinfo.value)

    @pytest.mark.parametrize("order", [1, 0.1, [0.1, 0.1], [10, 20], (100), (10, 10)])
    def test_rand_polarization_field_order_instance(
        self, order: int | list, mocker, polarization: Polarization, field_kwargs: dict
    ) -> None:
        shape = [16, 16]
        field_kwargs["order"] = order

        mock_gaussian_filter = mocker.patch(
            "pyvisgen.simulation.visibility.scipy.ndimage.gaussian_filter",
            return_value=np.ones(shape),
        )

        polarization.rand_polarization_field(shape, **field_kwargs)

        if isinstance(order, int | float):
            order = [order, order]

        sigma = torch.mean(torch.tensor(shape).double()) / (40 * torch.tensor(order))

        _, called_kwargs = mock_gaussian_filter.call_args
        np.testing.assert_allclose(called_kwargs["sigma"], sigma.numpy())

        # assert im. == torch.Size(order)

    def test_rand_polarization_field_order_wrong_dim(
        self, polarization: Polarization, field_kwargs: dict
    ) -> None:
        field_kwargs["order"] = [1, 16, 16]

        with pytest.raises(ValueError) as excinfo:
            polarization.rand_polarization_field([16, 16], **field_kwargs)

        assert "Expected len of 'order' to be 2!" in str(excinfo.value)

    def test_rand_polarization_field_scale_list(
        self, mocker, polarization: Polarization, field_kwargs: dict
    ) -> None:
        shape = [16, 16]
        field_kwargs["scale"] = [0.1, 1]

        mock_linspace = mocker.patch(
            "pyvisgen.simulation.visibility.torch.linspace",
            return_value=torch.linspace(*field_kwargs["scale"], np.prod(shape)),
        )
        polarization.rand_polarization_field(shape, **field_kwargs)

        called_args, _ = mock_linspace.call_args
        assert called_args == (*field_kwargs["scale"], np.prod(shape))

    def test_rand_polarization_field_scale_wrong_dim(
        self, polarization: Polarization, field_kwargs: dict
    ) -> None:
        field_kwargs["scale"] = [1, 0.1, 1]

        with pytest.raises(ValueError) as excinfo:
            polarization.rand_polarization_field([16, 16], **field_kwargs)

        assert "Expected len of 'scale' to be 2!" in str(excinfo.value)

    @pytest.mark.parametrize("threshold", [1e-4, 0.1, 1, 10])
    def test_rand_polarization_field_threshold(
        self, threshold: int | float, polarization: Polarization, field_kwargs: dict
    ) -> None:
        field_kwargs["threshold"] = threshold

        im = polarization.rand_polarization_field([16, 16], **field_kwargs)

        assert (im < threshold).all()

    @pytest.mark.parametrize("pol_type", ["linear", "circular"])
    def test_dop(self, pol_type: str, polarization_data: dict) -> None:
        _, H, W = polarization_data["SI"].shape

        polarization_data["polarization"] = pol_type
        pol = Polarization(**polarization_data)

        assert not hasattr(pol, "lin_dop")
        assert not hasattr(pol, "circ_dop")

        pol.dop()

        assert hasattr(pol, "lin_dop")
        assert hasattr(pol, "circ_dop")

        assert pol.lin_dop.shape == torch.Size([H, W])

    def test_stokes_matrix_no_polarization(
        self, mocker, polarization: Polarization
    ) -> None:
        mock_linear = mocker.patch.object(polarization, "linear")
        mock_circular = mocker.patch.object(polarization, "circular")

        B, mask, lin_dop, circ_dop = polarization.stokes_matrix()

        assert not mock_linear.called
        assert not mock_circular.called

        assert isinstance(B, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert isinstance(lin_dop, torch.Tensor)
        assert isinstance(circ_dop, torch.Tensor)

        assert B.device.type == polarization.device.type

        assert hasattr(polarization, "polarization_field")
        np.testing.assert_array_equal(
            polarization.polarization_field, torch.ones_like(polarization.I[..., 0])
        )

    def test_stokes_matrix_linear(self, mocker, polarization_data: dict) -> None:
        polarization_data["polarization"] = "linear"

        mock_linear = mocker.patch("pyvisgen.simulation.visibility.Polarization.linear")
        mock_circular = mocker.patch(
            "pyvisgen.simulation.visibility.Polarization.circular"
        )

        pol = Polarization(**polarization_data)
        B, mask, lin_dop, circ_dop = pol.stokes_matrix()

        assert mock_linear.called
        assert not mock_circular.called

        assert isinstance(B, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert isinstance(lin_dop, torch.Tensor)
        assert isinstance(circ_dop, torch.Tensor)

        assert B.device.type == pol.device.type

    def test_stokes_matrix_circular(self, mocker, polarization_data: dict) -> None:
        polarization_data["polarization"] = "circular"

        mock_linear = mocker.patch("pyvisgen.simulation.visibility.Polarization.linear")
        mock_circular = mocker.patch(
            "pyvisgen.simulation.visibility.Polarization.circular"
        )

        pol = Polarization(**polarization_data)
        B, mask, lin_dop, circ_dop = pol.stokes_matrix()

        assert not mock_linear.called
        assert mock_circular.called

        assert isinstance(B, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert isinstance(lin_dop, torch.Tensor)
        assert isinstance(circ_dop, torch.Tensor)

        assert B.device.type == pol.device.type


class TestVisLoop:
    @pytest.fixture
    def vis_args(self, obs, sky_dist: torch.Tensor) -> dict:
        return {"obs": obs, "SI": sky_dist}

    @pytest.fixture
    def mock_batch_loop(self, mocker, visibilities: Visibilities):
        return mocker.patch(
            "pyvisgen.simulation.visibility._batch_loop", return_value=visibilities
        )

    def test_torch_setup(
        self,
        mocker,
        vis_args: dict,
        mock_batch_loop,
    ) -> None:
        mock_torch_set_num_threads = mocker.patch(
            "pyvisgen.simulation.visibility.torch.set_num_threads"
        )

        mock_batch_loop  # noqa: B018
        vis_loop(**vis_args)

        mock_torch_set_num_threads.assert_called_with(10)
        assert torch._dynamo.config.suppress_errors is True

    @pytest.mark.parametrize("batch_size", [1, 10, 1000, int(1e8), "auto"])
    def test_batch_size(
        self, batch_size: int | str, vis_args: dict, mock_batch_loop
    ) -> None:
        mock_bl = mock_batch_loop
        vis_loop(**vis_args, batch_size=batch_size)

        call_args, _ = mock_bl.call_args

        if batch_size == "auto":
            obs = vis_args["obs"]
            bas = obs.baselines.get_valid_subset(obs.num_baselines, obs.device)
            batch_size = bas.baseline_nums.shape[0]

        assert call_args[0] == batch_size

    @pytest.mark.parametrize(
        "batch_size", [0.1, 10.0, [100], [1000.0], (10000,), "invalid"]
    )
    def test_batch_size_invalid(
        self, batch_size: float | list, vis_args: dict, mock_batch_loop
    ) -> None:
        mock_batch_loop  # noqa: B018

        with pytest.raises(ValueError) as excinfo:
            vis_loop(**vis_args, batch_size=batch_size)

        assert "Expected batch_size to be 'auto' or type int" in str(excinfo.value)

    def test_mode_full(self, mocker, vis_args: dict, mock_batch_loop) -> None:
        mock_batch_loop  # noqa: B018
        mock_get_valid_subset = mocker.patch.object(
            vis_args["obs"].baselines, "get_valid_subset"
        )

        vis_loop(**vis_args, mode="full")

        obs = vis_args["obs"]
        mock_get_valid_subset.assert_called_with(obs.num_baselines, obs.device)

    def test_mode_grid(self, mocker, vis_args: dict, mock_batch_loop, subset) -> None:
        mock_batch_loop  # noqa: B018

        mock_get_unique_grid = mocker.patch(
            "pyvisgen.simulation.observation.ValidBaselineSubset.get_unique_grid"
        )

        vis_loop(**vis_args, mode="grid")

        obs = vis_args["obs"]
        mock_get_unique_grid.assert_called_with(
            obs.fov, obs.ref_frequency, obs.img_size, obs.device
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mode_dense(self, mocker, vis_args: dict, mock_batch_loop, subset) -> None:
        mock_batch_loop  # noqa: B018
        mock_calc_dense_baselines = mocker.patch.object(
            vis_args["obs"], "calc_dense_baselines"
        )
        vis_args["obs"].dense_baselines_gpu = subset

        vis_loop(**vis_args, mode="dense")

        assert mock_calc_dense_baselines.called

    def test_mode_dense_raises_on_cpu(self, vis_args: dict, mock_batch_loop) -> None:
        # just in case dense does not raise an exception, we skip
        # the actual _batch_loop call
        mock_batch_loop  # noqa: B018

        vis_args["obs"].device = torch.device("cpu")

        with pytest.raises(ValueError) as excinfo:
            vis_loop(**vis_args, mode="dense")

        assert "Mode 'dense' is only available for GPU calculations!" in str(
            excinfo.value
        )

    def test_mode_invalid(self, vis_args: dict, mock_batch_loop) -> None:
        mode = "this_mode_is_invalid"
        with pytest.raises(ValueError) as excinfo:
            vis_loop(**vis_args, mode=mode)

        assert f"Unsupported mode: {mode}" in str(excinfo.value)
