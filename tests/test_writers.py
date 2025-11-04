from pathlib import Path

import numpy as np

from pyvisgen.io import PTWriter


output_path = Path("./tests/build/pt")
output_path.mkdir(parents=True)


def test_pt_writer_amp_phase():
    rng = np.random.default_rng(42)

    with PTWriter(
        output_path=output_path, dataset_type="train", amp_phase=True
    ) as writer:
        x_data = rng.uniform(size=(5, 10, 2, 256, 256))
        y_data = rng.uniform(size=(5, 10, 2, 256, 256))

        for bundle_id, (x, y) in enumerate(zip(x_data, y_data)):
            writer.write(x, y, index=bundle_id, bundle_length=len(x_data))


def test_pt_writer_real_imag():
    rng = np.random.default_rng(42)

    with PTWriter(
        output_path=output_path, dataset_type="train", amp_phase=False
    ) as writer:
        x_data = rng.uniform(size=(5, 10, 2, 256, 256))
        y_data = rng.uniform(size=(5, 10, 2, 256, 256))

        for bundle_id, (x, y) in enumerate(zip(x_data, y_data)):
            writer.write(x, y, index=bundle_id, bundle_length=len(x_data))
