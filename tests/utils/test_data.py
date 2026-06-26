import h5py
import numpy as np
import pytest

from pyvisgen.utils import load_bundles, open_bundles


@pytest.fixture
def create_h5(tmp_path) -> None:
    rng = np.random.default_rng()

    x = rng.uniform(size=(20, 32, 32))
    output_dir = tmp_path / "test_bundles"

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    for split in ["train", "valid", "test"]:
        with h5py.File(output_dir / f"{split}_data_0.h5", "w") as hf:
            hf.create_dataset("x", data=x[:10])

        with h5py.File(output_dir / f"{split}_data_1.h5", "w") as hf:
            hf.create_dataset("x", data=x[10:])


class TestLoadBundles:
    def test_load_bundles(self, tmp_path, create_h5) -> None:
        create_h5  # noqa: B018

        for split in ["train", "valid", "test"]:
            bundles = load_bundles(tmp_path / "test_bundles", dataset_type=split)

            assert len(bundles) == 2
            assert all(split in p for p in list(map(str, bundles)))

    def test_load_bundles_data_path_is_str(self, tmp_path, create_h5) -> None:
        create_h5  # noqa: B018

        for split in ["train", "valid", "test"]:
            bundles = load_bundles(str(tmp_path / "test_bundles"), dataset_type=split)

            assert len(bundles) == 2
            assert all(split in p for p in list(map(str, bundles)))


def test_open_bundles(tmp_path, create_h5) -> None:
    create_h5  # noqa: B018

    bundle = open_bundles(tmp_path / "test_bundles/train_data_0.h5", key="x")

    assert bundle.shape == (10, 32, 32)
