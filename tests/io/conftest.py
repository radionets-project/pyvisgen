from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    output = tmp_path / "output"

    if not output.exists():
        output.mkdir(parents=True)

    return output


@pytest.fixture
def data_sample() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng()

    x = rng.uniform(size=(10, 2, 32, 32))
    y = rng.uniform(size=(10, 2, 32, 32))

    return x, y
