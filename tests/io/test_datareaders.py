from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from pyvisgen.io.datareaders import UVH5Data, UVH5Reader, read_uvh5
from pyvisgen.io.datawriters import UVH5Writer
from pyvisgen.simulation import Visibilities

TEST_LAYOUT = Path(__file__).parents[1] / "data" / "test_layout.txt"


@pytest.fixture
def uvh5_file(output_dir, uvh5_vis_data, uvh5_obs, uvh5_sky) -> Path:
    with UVH5Writer(output_path=output_dir, dataset_type="train") as writer:
        writer.write(uvh5_vis_data, uvh5_obs, index=0, sky=uvh5_sky)

    return output_dir / "train_0.uvh5"


class TestReadUVH5:
    """This tests the *function* read_uvh5."""

    def test_returns_visibilities(self, uvh5_file, uvh5_vis_data) -> None:
        data = read_uvh5(uvh5_file)

        assert isinstance(data, UVH5Data)
        assert isinstance(data.vis, Visibilities)
        assert data.path == uvh5_file

        for name in ("V_11", "V_22", "V_12", "V_21", "weights", "u", "v", "w"):
            assert torch.equal(getattr(data.vis, name), getattr(uvh5_vis_data, name))

        assert torch.equal(data.vis.date, uvh5_vis_data.date)
        assert torch.equal(data.vis.st_id_pairs, uvh5_vis_data.st_id_pairs)

    def test_rebuilt_fields(self, uvh5_file, uvh5_vis_data) -> None:
        vis = read_uvh5(uvh5_file).vis
        st = uvh5_vis_data.st_id_pairs
        n = st.shape[0]

        assert torch.equal(vis.num, torch.arange(1, n + 1, dtype=torch.float64))
        assert torch.equal(
            vis.base_num, (256 * (st[:, 0] + 1) + st[:, 1] + 1).to(torch.float64)
        )
        assert vis.linear_dop.numel() == 0
        assert vis.circular_dop.numel() == 0

    def test_get_values_matches_simulation_layout(
        self, uvh5_file, uvh5_vis_data
    ) -> None:
        values = read_uvh5(uvh5_file).vis.get_values()

        assert values.shape == (*uvh5_vis_data.V_11.shape, 4)
        assert torch.equal(values[..., 0], uvh5_vis_data.V_11)
        assert torch.equal(values[..., 3], uvh5_vis_data.V_21)

    def test_observation(self, uvh5_file, uvh5_obs, uvh5_vis_data) -> None:
        obs = read_uvh5(uvh5_file).obs

        assert torch.equal(obs.ra, uvh5_obs.ra)
        assert torch.equal(obs.dec, uvh5_obs.dec)
        assert obs.ref_frequency == uvh5_obs.ref_frequency
        assert torch.allclose(obs.frequency_offsets, uvh5_obs.frequency_offsets)
        assert torch.equal(obs.bandwidths, uvh5_obs.bandwidths)
        assert obs.layout == uvh5_obs.layout
        assert torch.equal(obs.lm, uvh5_obs.lm)
        assert obs.normalize is True

        subset = obs.baselines.get_valid_subset(num_baselines=123, device="cpu")
        assert torch.equal(subset.date, uvh5_vis_data.date)

    def test_sky_and_img_size(self, uvh5_file, uvh5_sky) -> None:
        data = read_uvh5(uvh5_file)

        assert torch.equal(data.sky, uvh5_sky)  # ty: ignore[invalid-argument-type]
        assert data.img_size == uvh5_sky.shape[-1]

    def test_without_sky(self, output_dir, uvh5_vis_data, uvh5_obs) -> None:
        with UVH5Writer(output_path=output_dir, dataset_type="test") as writer:
            writer.write(uvh5_vis_data, uvh5_obs, index=0)

        data = read_uvh5(output_dir / "test_0.uvh5")

        assert data.sky is None
        assert data.img_size is None

    def test_layout_override(self, uvh5_file) -> None:
        obs = read_uvh5(uvh5_file, layout=TEST_LAYOUT).obs

        assert obs.layout == TEST_LAYOUT
        assert len(obs.array.st_num) == 10
        assert obs.num_baselines == 45
        assert len(obs.array_earth_loc) == 10

    def test_stored_layout_path_is_loaded(
        self, output_dir, uvh5_vis_data, uvh5_obs
    ) -> None:
        uvh5_obs.layout = str(TEST_LAYOUT)
        with UVH5Writer(output_path=output_dir, dataset_type="custom") as writer:
            writer.write(uvh5_vis_data, uvh5_obs, index=0)

        obs = read_uvh5(output_dir / "custom_0.uvh5").obs

        assert obs.layout == str(TEST_LAYOUT)
        assert len(obs.array.st_num) == 10

    def test_num_baselines_without_loadable_layout(self, uvh5_file) -> None:
        obs = read_uvh5(uvh5_file, layout="no_such_layout").obs

        # the fixture has 16 distinct station ids
        assert obs.num_baselines == 16 * 15 // 2

    def test_missing_layout_raises_on_array_access(self, uvh5_file) -> None:
        with h5py.File(uvh5_file, "a") as f:
            del f["obs"]["layout"]

        obs = read_uvh5(uvh5_file).obs

        assert obs.layout is None
        with pytest.raises(ValueError, match="no array layout"):
            _ = obs.array

    def test_missing_obs_group_raises(self, uvh5_file) -> None:
        with h5py.File(uvh5_file, "a") as f:
            del f["obs"]

        with pytest.raises(KeyError, match="no 'obs' group"):
            read_uvh5(uvh5_file)


class TestUVH5Reader:
    """This tests the *class* UVH5Reader."""

    def test_single_file(self, uvh5_file) -> None:
        reader = UVH5Reader(uvh5_file)

        assert len(reader) == 1
        assert reader[0].path == uvh5_file

    def test_layout_passed_through(self, uvh5_file) -> None:
        assert UVH5Reader(uvh5_file, layout=TEST_LAYOUT)[0].obs.layout == TEST_LAYOUT

    def test_missing_path_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            UVH5Reader(tmp_path / "nope")

    def test_empty_directory_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="No .uvh5 files"):
            UVH5Reader(tmp_path)


class TestGridderIntegration:
    def test_from_pyvisgen_accepts_reader_output(self, uvh5_file) -> None:
        gridder_module = pytest.importorskip("pyvisgrid.core.gridder")

        data = read_uvh5(uvh5_file, layout=TEST_LAYOUT)
        grid_data = gridder_module.Gridder.from_pyvisgen(
            vis_data=data.vis,
            obs=data.obs,
            img_size=data.img_size,
            fov=200,
            stokes_components="I",
            polarizations=None,
        ).grid()

        mask_real_imag = np.array(grid_data.get_mask_real_imag())
        assert mask_real_imag.shape[-2:] == (data.img_size, data.img_size)
        assert np.isfinite(mask_real_imag).all()
