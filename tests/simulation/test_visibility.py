from dataclasses import fields

import numpy as np
import pytest
import torch

from pyvisgen.simulation.visibility import (
    Polarization,
    Visibilities,
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
