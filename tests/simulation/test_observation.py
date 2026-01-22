from dataclasses import fields

import numpy as np
import pytest
import torch

from pyvisgen.simulation.observation import Baselines, ValidBaselineSubset


class TestBaselines:
    def test_get_item(self, baselines: Baselines) -> None:
        idx = 0

        baselines_item = baselines[idx]

        assert isinstance(baselines_item, Baselines)

        for f in fields(Baselines):
            assert getattr(baselines_item, f.name) == getattr(baselines, f.name)[idx]

    def test_get_slice(self, baselines: Baselines) -> None:
        baselines_items = baselines[2:-2]

        assert isinstance(baselines_items, Baselines)
        assert baselines_items.u.shape == torch.Size([6])

    def test_add_baseline(
        self, baselines: Baselines, baselines_data: dict[str, torch.Tensor]
    ) -> None:
        orig_len = baselines.u.shape[0]
        new_baselines = Baselines(**baselines_data)

        baselines.add_baseline(new_baselines)

        assert (
            baselines.u.shape[0] == orig_len + new_baselines.u.shape[0]
        )  # torch.Size([20])

    def test_get_valid_subset(self, baselines: Baselines, device: str) -> None:
        valid = baselines.valid

        subset = baselines.get_valid_subset(len(baselines.time), device=device)

        assert isinstance(subset, ValidBaselineSubset)
        assert len(subset.date) == sum(valid)


class TestValidBaselineSubset:
    def test_get_item(self, subset: ValidBaselineSubset) -> None:
        idx = 0

        subset_item = subset[idx]

        assert isinstance(subset_item, ValidBaselineSubset)

        for f in fields(ValidBaselineSubset):
            assert getattr(subset_item, f.name) == getattr(subset, f.name)[idx]

    def test_get_slice(self, subset: ValidBaselineSubset) -> None:
        subset_items = subset[2:-2]

        assert isinstance(subset_items, ValidBaselineSubset)
        assert subset_items.u_start.shape == torch.Size([6])

    def test_get_timerange(self, subset: ValidBaselineSubset, subset_data) -> None:
        dates = subset_data["date"]

        subset_items = subset.get_timerange(t_start=dates[2], t_stop=dates[-3])

        assert isinstance(subset_items, ValidBaselineSubset)
        assert subset_items.u_start.shape == torch.Size([6])
        np.testing.assert_array_equal(
            subset_items.date.detach().cpu(), dates[2:-2].detach().cpu()
        )

    @pytest.mark.parametrize(
        "fov,freq,img_size,expected",
        [
            (0.1, 15e9, 32, 8),
            (0.1, 15e9, 64, 10),
            (3600, 1.4e9, 32, 2),
            (0.1, 15e9, 35, 8),  # also test for odd image size
            (1e-8, 1.4e9, 32, 1),  # very small fov
        ],
    )
    def test_get_unique_grid(
        self,
        fov: float,
        freq: float,
        img_size: int,
        expected: int,
        subset_data: dict[str, torch.Tensor],
        device: str,
    ) -> None:
        size = 10
        dev = torch.device(device)

        subset_data["u_valid"] = torch.linspace(-1e6, 1e6, size, device=dev)
        subset_data["v_valid"] = torch.linspace(-1e6, 1e6, size, device=dev)

        subset = ValidBaselineSubset(**subset_data)

        unique = subset.get_unique_grid(
            fov=fov, ref_frequency=freq, img_size=img_size, device=device
        )

        assert isinstance(unique, ValidBaselineSubset)
        assert unique.u_valid.shape == torch.Size([expected])
