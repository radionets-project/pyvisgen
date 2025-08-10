import pytest
import torch

from pyvisgen.simulation.scan import (
    angular_distance,
    calc_beam,
    calc_feed_rotation,
    calc_fourier,
    integrate,
    jinc,
    rime,
)


# This is added to mock torch.compile
@pytest.fixture(autouse=True)
def disable_torch_compile(monkeypatch):
    """Disable torch.compile to make tests work properly."""

    def identity(func, *args, **kwargs):
        """Return the function unchanged."""
        return func

    # Replace torch.compile with the identity function
    monkeypatch.setattr(torch, "compile", identity)


@pytest.fixture
def setup_test_data():
    # Set deterministic behavior
    torch.manual_seed(42)

    # Create a simple 2x2 image with a single source in the center
    img = torch.zeros(1, 2, 2, dtype=torch.complex128)
    img[0, 0, 0] = 1.0 + 0.5j
    img[0, 0, 1] = 1.0 + 0.5j

    # Create a simple 2D lm grid
    lm = torch.zeros(1, 1, 2, dtype=torch.float64)
    lm[..., 0] = torch.tensor([0.01])
    lm[..., 1] = torch.tensor([0.01])
    lm = lm.flatten(end_dim=1)

    # Simulate baselines dataclass as a list for testing
    # [0:u1, 1:u2, 2:u_valid, 3:v1, 4:v2, 5:v_valid,
    #  6:w1, 7:w2, 8:w_valid, 9:t1, 10:t2, 11:t_valid,
    #  12:q1a, 13:q1b, 14:q1ab_valid, 15:q2a, 16:q2b, 17:q2ab_valid]
    bas = [
        None,
        None,
        torch.tensor([100.0, 200.0, 300.0]),  # u
        None,
        None,
        torch.tensor([150.0, 250.0, 350.0]),  # v
        None,
        None,
        torch.tensor([50.0, 100.0, 150.0]),  # w
        None,
        None,
        None,
        None,
        torch.tensor([0.1]),  # q1
        None,
        None,
        torch.tensor([0.15]),  # q2
        None,
    ]

    # Sky position
    ra = torch.tensor(0.0)
    dec = torch.tensor(0.0)

    # Antenna properties
    ant_diam = torch.tensor([25.0])

    # Spectral window frequencies
    spw_low = 1.0e9  # 1 GHz
    spw_high = 2.0e9  # 2 GHz

    # For angular distance
    rd = torch.zeros(1, 1, 2, dtype=torch.float64)
    rd[..., 0] = 0.001  # ra
    rd[..., 1] = 0.001  # dec
    rd = rd.flatten(end_dim=1)

    return {
        "img": img,
        "bas": bas,
        "lm": lm,
        "rd": rd,
        "ra": ra,
        "dec": dec,
        "ant_diam": ant_diam,
        "spw_low": spw_low,
        "spw_high": spw_high,
        "polarization": "circular",
    }


class TestScan:
    """Unit tests for pyvisgen.simulation.scan module."""

    def test_jinc(self):
        """Test the jinc function."""
        # Test with zero
        x_zero = torch.tensor([0.0], dtype=torch.float64)
        result_zero = jinc(x_zero)
        assert torch.isclose(result_zero, torch.tensor([1.0], dtype=torch.float64))

        # Test with standard values
        x_values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = jinc(x_values)

        # Calculate expected values directly from torch's bessel_j1 function
        expected = torch.zeros_like(x_values, dtype=torch.float64)
        expected[x_values != 0] = (
            2
            * torch.special.bessel_j1(x_values[x_values != 0])
            / x_values[x_values != 0]
        )

        assert torch.allclose(result, expected, rtol=1e-6)

        # Additional test to verify actual values
        # Compare with actual values from the executed function
        assert torch.isclose(
            result[0], torch.tensor(0.8801, dtype=torch.float64), rtol=1e-4
        )
        assert torch.isclose(
            result[1], torch.tensor(0.5767, dtype=torch.float64), rtol=1e-4
        )
        assert torch.isclose(
            result[2], torch.tensor(0.2260, dtype=torch.float64), rtol=1e-3
        )

    def test_angular_distance(self, setup_test_data):
        """Test the angular_distance function."""
        rd = setup_test_data["rd"]
        ra = setup_test_data["ra"]
        dec = setup_test_data["dec"]

        result = angular_distance(rd, ra, dec)

        # We expect theta to be arcsin(sqrt(rd[...,0]²+rd[...,1]²)) for each point
        expected_shape = (1,)
        assert result.shape == expected_shape

        # For our values (0.001, 0.001), the angular distance is approximately 0.0014
        expected_value = torch.arcsin(
            torch.sqrt(torch.tensor(0.001) ** 2 + torch.tensor(0.001) ** 2)
        )
        assert torch.allclose(result, expected_value, rtol=1e-5)

    def test_calc_fourier(self, setup_test_data):
        """Test the calc_fourier function."""
        img = setup_test_data["img"]
        bas = setup_test_data["bas"]
        lm = setup_test_data["lm"].flatten(end_dim=-2)
        spw_low = setup_test_data["spw_low"]
        spw_high = setup_test_data["spw_high"]

        X1, X2 = calc_fourier(img, img, bas, lm, spw_low, spw_high)

        # Check shapes
        assert X1.shape[-2:] == img.shape[-2:]
        assert X2.shape[-2:] == img.shape[-2:]
        assert img.real[img.real > 0].shape == lm[-1].shape

        # Check types
        assert X1.dtype == torch.complex128
        assert X2.dtype == torch.complex128

        # Check values are different for different frequencies
        assert not torch.allclose(X1, X2)

    def test_calc_feed_rotation_circular(self, setup_test_data):
        """Test the feed rotation calculation for circular polarization."""
        # Simulate polarized visibilities with 2x2 Jones matrices
        X1 = torch.ones(2, 1, 2, 2, dtype=torch.complex128)
        X2 = torch.ones(2, 1, 2, 2, dtype=torch.complex128)
        bas = setup_test_data["bas"]
        polarization = "circular"

        X1_rot, X2_rot = calc_feed_rotation(X1, X2, bas, polarization)

        # Check shape preservation
        assert X1_rot.shape == X1.shape
        assert X2_rot.shape == X2.shape

        # Check types
        assert X1_rot.dtype == torch.complex128
        assert X2_rot.dtype == torch.complex128

        # For circular polarization, values should be modified by complex rotation
        assert not torch.allclose(X1_rot, X1)
        assert not torch.allclose(X2_rot, X2)

    def test_calc_feed_rotation_linear(self, setup_test_data):
        """Test the feed rotation calculation for linear polarization."""
        # Simulate polarized visibilities with 2x2 Jones matrices
        X1 = torch.ones(2, 1, 2, 2, dtype=torch.complex128)
        X2 = torch.ones(2, 1, 2, 2, dtype=torch.complex128)
        bas = setup_test_data["bas"]
        polarization = "linear"

        X1_rot, X2_rot = calc_feed_rotation(X1, X2, bas, polarization)

        # Check shape preservation
        assert X1_rot.shape == X1.shape
        assert X2_rot.shape == X2.shape

        # Check types
        assert X1_rot.dtype == torch.complex128
        assert X2_rot.dtype == torch.complex128

        # For linear polarization, values should be modified by trigonometric functions
        assert not torch.allclose(X1_rot, X1)
        assert not torch.allclose(X2_rot, X2)

    def test_calc_beam(self, setup_test_data):
        """Test the beam calculation function."""
        # Create test data for the beam calculation
        X1 = torch.ones(2, 1, 2, 2, dtype=torch.complex128)
        X2 = torch.ones(2, 1, 2, 2, dtype=torch.complex128)
        rd = setup_test_data["rd"]
        ra = setup_test_data["ra"]
        dec = setup_test_data["dec"]
        ant_diam = setup_test_data["ant_diam"]
        spw_low = setup_test_data["spw_low"]
        spw_high = setup_test_data["spw_high"]

        EXE1, EXE2 = calc_beam(X1, X2, rd, ra, dec, ant_diam, spw_low, spw_high)

        # Check shape preservation
        assert EXE1.shape == X1.shape
        assert EXE2.shape == X2.shape

        # Check types
        assert EXE1.dtype == torch.complex128
        assert EXE2.dtype == torch.complex128

        # Check values are different for beam-corrected data
        assert not torch.allclose(EXE1, X1)
        assert not torch.allclose(EXE2, X2)

    def test_integrate(self, setup_test_data):
        """Test the integration function."""
        # Create test data for integration
        X1 = torch.ones(2, 2, 2, 2, dtype=torch.complex128)
        X2 = torch.ones(2, 2, 2, 2, dtype=torch.complex128)

        result = integrate(X1, X2)

        # Expected shape is (2, 2, 2) after summing over one dimension (lm)
        # and averaging over frequency
        expected_shape = (2, 2, 2)
        # Because X1 has all 1s and X2 has all 2s, average should be 1.5
        expected_value = 0.5 * 4  # 4 elements summed for each entry

        # Check shape
        assert result.shape == expected_shape

        # Check that values match expectation
        assert torch.allclose(
            result.real,
            torch.full(expected_shape, expected_value.real, dtype=torch.float64),
        )

    def test_rime(self, setup_test_data):
        """Test the complete RIME function."""
        # Extract all needed parameters
        img = setup_test_data["img"]
        bas = setup_test_data["bas"]
        lm = setup_test_data["lm"]
        rd = setup_test_data["rd"]
        ra = setup_test_data["ra"]
        dec = setup_test_data["dec"]
        ant_diam = setup_test_data["ant_diam"]
        spw_low = setup_test_data["spw_low"]
        spw_high = setup_test_data["spw_high"]
        polarization = setup_test_data["polarization"]

        # Test with mode = "grid" (default case)
        vis_grid = rime(
            img,
            bas,
            lm,
            rd,
            ra,
            dec,
            ant_diam,
            spw_low,
            spw_high,
            polarization,
            mode="grid",
            corrupted=True,
        )

        # Test with mode = "grid" (reversed jones ordering)
        vis_grid_reversed = rime(
            img,
            bas,
            lm,
            rd,
            ra,
            dec,
            ant_diam,
            spw_low,
            spw_high,
            polarization,
            mode="grid",
            corrupted=True,
            ft="reversed",
        )

        # Test with mode = "grid" (reversed jones ordering, no polarization)
        vis_grid_reversed_nopol = rime(
            img,
            bas,
            lm,
            rd,
            ra,
            dec,
            ant_diam,
            spw_low,
            spw_high,
            polarization=None,
            mode="grid",
            corrupted=True,
            ft="reversed",
        )

        # Test with mode = "grid" (use radioft dft)
        vis_grid_dft = rime(
            img,
            bas,
            lm,
            rd,
            ra,
            dec,
            ant_diam,
            spw_low,
            spw_high,
            polarization,
            mode="grid",
            corrupted=True,
            ft="dft",
        )

        assert torch.isclose(vis_grid_reversed, vis_grid, rtol=1e-8).all()
        assert torch.isclose(vis_grid_reversed, vis_grid, rtol=1e-8).all()
        # assert torch.isclose(vis_grid, vis_grid_dft.cpu(), rtol=1e-8).all()
        assert torch.isclose(
            vis_grid_reversed_nopol, vis_grid_dft.cpu(), rtol=1e-8
        ).all()
        assert vis_grid_reversed.dtype == vis_grid.dtype
        assert vis_grid_reversed.shape == vis_grid.shape

        # Test with mode = "grid" with polarization
        vis_grid_pol = rime(
            img, bas, lm, rd, ra, dec, ant_diam, spw_low, spw_high, polarization, "grid"
        )

        # Check output shape, should be (baseline, 2, 2)
        expected_shape = (3, 2, 2)
        assert vis_grid_pol.shape == expected_shape

        # Check type
        assert vis_grid_pol.dtype == torch.complex128

        # Test with corrupted=True
        vis_corrupted_pol = rime(
            img,
            bas,
            lm,
            rd,
            ra,
            dec,
            ant_diam,
            spw_low,
            spw_high,
            polarization,
            "grid",
            corrupted=True,
        )

        # Shape should be the same, but values should differ
        assert vis_corrupted_pol.shape == expected_shape
        assert not torch.allclose(vis_corrupted_pol, vis_grid_pol)
