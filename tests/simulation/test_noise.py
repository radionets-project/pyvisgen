import numpy as np
import pytest
import torch

from pyvisgen.simulation.noise import (
    _TELESCOPES,
    _interp1d,
    compute_noise_std,
    elevation_tsys_contribution,
    generate_noise,
    sefd_from_elevation,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def n_baselines() -> int:
    return 16


@pytest.fixture(scope="module")
def el_deg(n_baselines: int) -> torch.Tensor:
    """Per-baseline elevations spanning the MeerKAT table range."""
    return torch.linspace(20.0, 80.0, n_baselines)


@pytest.fixture(scope="module")
def tsys_over_eta() -> float:
    return 20.0  # K — typical MeerKAT value


# ---------------------------------------------------------------------------
# _interp1d
# ---------------------------------------------------------------------------


class TestInterp1d:
    @pytest.fixture(scope="class")
    def knots(self) -> tuple[torch.Tensor, torch.Tensor]:
        xp = torch.tensor([0.0, 1.0, 2.0, 3.0])
        fp = torch.tensor([0.0, 10.0, 20.0, 30.0])
        return xp, fp

    def test_exact_knot_values(self, knots) -> None:
        xp, fp = knots
        result = _interp1d(xp, xp, fp)
        np.testing.assert_allclose(result.numpy(), fp.numpy())

    def test_midpoint_interpolation(self, knots) -> None:
        xp, fp = knots
        x = torch.tensor([0.5, 1.5, 2.5])
        result = _interp1d(x, xp, fp)
        np.testing.assert_allclose(result.numpy(), [5.0, 15.0, 25.0])

    def test_clamping_below_min(self, knots) -> None:
        xp, fp = knots
        result = _interp1d(torch.tensor([-10.0]), xp, fp)
        np.testing.assert_allclose(result.numpy(), [fp[0].item()])

    def test_clamping_above_max(self, knots) -> None:
        xp, fp = knots
        result = _interp1d(torch.tensor([100.0]), xp, fp)
        np.testing.assert_allclose(result.numpy(), [fp[-1].item()])

    def test_output_shape_matches_input(self, knots) -> None:
        xp, fp = knots
        x = torch.rand(5, 3)
        result = _interp1d(x, xp, fp)
        assert result.shape == x.shape


# ---------------------------------------------------------------------------
# elevation_tsys_contribution
# ---------------------------------------------------------------------------


class TestElevationTsysContribution:
    @pytest.mark.parametrize("pol", ["H", "V", "mean"])
    def test_output_shape(self, el_deg: torch.Tensor, pol: str) -> None:
        result = elevation_tsys_contribution(el_deg, pol=pol)
        assert result.shape == el_deg.shape

    @pytest.mark.parametrize("pol", ["H", "V", "mean"])
    def test_positive_values(self, el_deg: torch.Tensor, pol: str) -> None:
        result = elevation_tsys_contribution(el_deg, pol=pol)
        assert (result > 0).all()

    def test_pol_mean_is_average_of_h_and_v(self, el_deg: torch.Tensor) -> None:
        t_h = elevation_tsys_contribution(el_deg, pol="H")
        t_v = elevation_tsys_contribution(el_deg, pol="V")
        t_mean = elevation_tsys_contribution(el_deg, pol="mean")
        np.testing.assert_allclose(t_mean.numpy(), ((t_h + t_v) / 2).numpy(), rtol=1e-6)

    def test_t_atm_decreases_with_elevation(self) -> None:
        """Atmospheric path is shorter at high elevation → lower T_atm."""
        el_low = torch.tensor([20.0])
        el_high = torch.tensor([80.0])
        assert elevation_tsys_contribution(el_low) > elevation_tsys_contribution(
            el_high
        )

    def test_unknown_telescope_raises(self, el_deg: torch.Tensor) -> None:
        with pytest.raises(KeyError):
            elevation_tsys_contribution(el_deg, telescope="unknown_telescope")

    @pytest.mark.parametrize("telescope", list(_TELESCOPES.keys()))
    def test_registered_telescopes(self, el_deg: torch.Tensor, telescope: str) -> None:
        result = elevation_tsys_contribution(el_deg, telescope=telescope)
        assert result.shape == el_deg.shape


# ---------------------------------------------------------------------------
# sefd_from_elevation
# ---------------------------------------------------------------------------


class TestSefdFromElevation:
    def test_output_shape(self, el_deg: torch.Tensor, tsys_over_eta: float) -> None:
        sefd = sefd_from_elevation(el_deg, el_deg, tsys_over_eta)
        assert sefd.shape == el_deg.shape

    def test_positive_values(self, el_deg: torch.Tensor, tsys_over_eta: float) -> None:
        sefd = sefd_from_elevation(el_deg, el_deg, tsys_over_eta)
        assert (sefd > 0).all()

    def test_lower_elevation_higher_sefd(self, tsys_over_eta: float) -> None:
        """Higher Tsys at low elevation → higher SEFD."""
        el_low = torch.tensor([20.0])
        el_high = torch.tensor([70.0])
        sefd_low = sefd_from_elevation(el_low, el_low, tsys_over_eta)
        sefd_high = sefd_from_elevation(el_high, el_high, tsys_over_eta)
        assert sefd_low > sefd_high

    def test_at_reference_elevation(self, tsys_over_eta: float) -> None:
        """At the reference elevation (55°) delta=0,
        so SEFD equals direct calculation.
        """
        k_B = 1.38e-23
        d = _TELESCOPES["meerkat"]["dish_diameter"]
        A_geom = torch.pi * (d / 2) ** 2
        expected = 2 * k_B * tsys_over_eta / A_geom * 1e26

        el_ref = torch.tensor([55.0])
        sefd = sefd_from_elevation(el_ref, el_ref, tsys_over_eta)
        np.testing.assert_allclose(sefd.item(), expected, rtol=1e-5)

    def test_geometric_mean_of_two_antennas(self, tsys_over_eta: float) -> None:
        """Baseline SEFD is the geometric mean of both antenna SEFDs."""
        k_B = 1.38e-23
        d = _TELESCOPES["meerkat"]["dish_diameter"]
        A_geom = torch.pi * (d / 2) ** 2

        el1 = torch.tensor([30.0])
        el2 = torch.tensor([60.0])

        ref_el = torch.tensor(55.0)
        t_ref = elevation_tsys_contribution(ref_el)
        delta1 = elevation_tsys_contribution(el1) - t_ref
        delta2 = elevation_tsys_contribution(el2) - t_ref
        sefd1 = 2 * k_B * (tsys_over_eta + delta1) / A_geom * 1e26
        sefd2 = 2 * k_B * (tsys_over_eta + delta2) / A_geom * 1e26
        expected = torch.sqrt(sefd1 * sefd2)

        sefd = sefd_from_elevation(el1, el2, tsys_over_eta)
        np.testing.assert_allclose(sefd.numpy(), expected.numpy(), rtol=1e-6)

    @pytest.mark.parametrize("telescope", list(_TELESCOPES.keys()))
    def test_registered_telescopes(
        self, el_deg: torch.Tensor, tsys_over_eta: float, telescope: str
    ) -> None:
        sefd = sefd_from_elevation(el_deg, el_deg, tsys_over_eta, telescope=telescope)
        assert sefd.shape == el_deg.shape
        assert (sefd > 0).all()


# ---------------------------------------------------------------------------
# compute_noise_std
# ---------------------------------------------------------------------------


class TestComputeNoiseStd:
    def test_output_shape(self, obs, n_baselines: int) -> None:
        sefd = torch.rand(n_baselines) * 1000
        std = compute_noise_std(obs, sefd)
        assert std.shape == sefd.shape

    def test_scalar_sefd(self, obs) -> None:
        sefd = torch.tensor(400.0)
        std = compute_noise_std(obs, sefd)
        assert std.shape == sefd.shape

    def test_larger_sefd_larger_std(self, obs) -> None:
        sefd_lo = torch.tensor(100.0)
        sefd_hi = torch.tensor(1000.0)
        assert compute_noise_std(obs, sefd_hi) > compute_noise_std(obs, sefd_lo)

    def test_radiometer_equation(self, obs) -> None:
        """Verify σ = SEFD / (η * sqrt(2 * Δν * τ))."""
        sefd = torch.tensor(400.0)
        eta = 0.93
        expected = sefd / (eta * torch.sqrt(2 * obs.int_time * obs.bandwidths[0]))
        result = compute_noise_std(obs, sefd)
        np.testing.assert_allclose(result.item(), expected.item(), rtol=1e-6)


# ---------------------------------------------------------------------------
# generate_noise
# ---------------------------------------------------------------------------


class TestGenerateNoise:
    @pytest.mark.parametrize(
        "n_baselines,n_channels",
        [(8, 1), (16, 4), (32, 1)],
    )
    def test_sefd_mode_output_shapes(
        self, n_baselines: int, n_channels: int, obs
    ) -> None:
        shape = (n_baselines, n_channels, 2, 2)
        noise, weights = generate_noise(shape, obs, noise_value=400.0, mode="sefd")
        assert noise.shape == torch.Size(shape)
        assert weights.shape == torch.Size([n_baselines, n_channels])

    def test_sefd_mode_complex(self, obs, n_baselines: int) -> None:
        shape = (n_baselines, 1, 2, 2)
        noise, _ = generate_noise(shape, obs, noise_value=400.0, mode="sefd")
        assert torch.is_complex(noise)

    def test_sefd_mode_weights_positive(self, obs, n_baselines: int) -> None:
        shape = (n_baselines, 1, 2, 2)
        _, weights = generate_noise(shape, obs, noise_value=400.0, mode="sefd")
        assert (weights > 0).all()

    def test_sefd_mode_no_elevations_required(self, obs, n_baselines: int) -> None:
        """SEFD mode must work without passing elevation tensors."""
        shape = (n_baselines, 1, 2, 2)
        noise, weights = generate_noise(shape, obs, noise_value=400.0, mode="sefd")
        assert noise.shape == torch.Size(shape)

    @pytest.mark.parametrize(
        "n_baselines,n_channels",
        [(8, 1), (16, 4), (32, 1)],
    )
    def test_tsys_mode_output_shapes(
        self, n_baselines: int, n_channels: int, obs, tsys_over_eta: float
    ) -> None:
        shape = (n_baselines, n_channels, 2, 2)
        el = torch.linspace(20.0, 80.0, n_baselines)
        noise, weights = generate_noise(
            shape, obs, noise_value=tsys_over_eta, mode="tsys", el1_deg=el, el2_deg=el
        )
        assert noise.shape == torch.Size(shape)
        assert weights.shape == torch.Size([n_baselines, n_channels])

    def test_tsys_mode_complex(
        self, obs, n_baselines: int, tsys_over_eta: float
    ) -> None:
        shape = (n_baselines, 1, 2, 2)
        el = torch.linspace(20.0, 80.0, n_baselines)
        noise, _ = generate_noise(
            shape, obs, noise_value=tsys_over_eta, mode="tsys", el1_deg=el, el2_deg=el
        )
        assert torch.is_complex(noise)

    def test_tsys_mode_weights_positive(
        self, obs, n_baselines: int, tsys_over_eta: float
    ) -> None:
        shape = (n_baselines, 1, 2, 2)
        el = torch.linspace(20.0, 80.0, n_baselines)
        _, weights = generate_noise(
            shape, obs, noise_value=tsys_over_eta, mode="tsys", el1_deg=el, el2_deg=el
        )
        assert (weights > 0).all()

    def test_tsys_mode_requires_elevations(self, obs, n_baselines: int) -> None:
        shape = (n_baselines, 1, 2, 2)
        with pytest.raises(ValueError, match="el1_deg and el2_deg are required"):
            generate_noise(shape, obs, noise_value=20.0, mode="tsys")

    def test_unknown_mode_raises(self, obs, n_baselines: int) -> None:
        shape = (n_baselines, 1, 2, 2)
        with pytest.raises(ValueError, match="Unknown noise mode"):
            generate_noise(shape, obs, noise_value=400.0, mode="invalid_mode")

    def test_unknown_telescope_raises(self, obs, n_baselines: int) -> None:
        shape = (n_baselines, 1, 2, 2)
        el = torch.linspace(20.0, 80.0, n_baselines)
        with pytest.raises(ValueError, match="Unknown telescope"):
            generate_noise(
                shape,
                obs,
                noise_value=20.0,
                mode="tsys",
                el1_deg=el,
                el2_deg=el,
                telescope="unknown_telescope",
            )

    def test_higher_sefd_higher_noise_std(self, obs, n_baselines: int) -> None:
        shape = (n_baselines, 1, 2, 2)
        torch.manual_seed(0)
        noise_lo, weights_lo = generate_noise(
            shape, obs, noise_value=100.0, mode="sefd"
        )
        torch.manual_seed(0)
        noise_hi, weights_hi = generate_noise(
            shape, obs, noise_value=10000.0, mode="sefd"
        )
        assert noise_hi.real.std() > noise_lo.real.std()
        assert weights_lo.mean() > weights_hi.mean()

    def test_tsys_weights_vary_with_elevation(
        self, obs, n_baselines: int, tsys_over_eta: float
    ) -> None:
        shape = (n_baselines, 1, 2, 2)
        el = torch.linspace(20.0, 80.0, n_baselines)
        _, weights = generate_noise(
            shape, obs, noise_value=tsys_over_eta, mode="tsys", el1_deg=el, el2_deg=el
        )
        # weights should not all be equal since elevations span a range
        assert not torch.allclose(weights[0], weights[-1])

    def test_noise_mean_near_zero(self, obs) -> None:
        """Real and imaginary parts should have mean ≈ 0 for large sample count."""
        torch.manual_seed(42)
        shape = (2048, 1, 2, 2)
        noise, _ = generate_noise(shape, obs, noise_value=400.0, mode="sefd")
        np.testing.assert_allclose(noise.real.mean().item(), 0.0, atol=0.5)
        np.testing.assert_allclose(noise.imag.mean().item(), 0.0, atol=0.5)
