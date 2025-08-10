import shutil

import pytest
import torch

# Replace torch.compile with a function that returns the original function unchanged
original_compile = torch.compile


def mock_compile(fn, **kwargs):
    return fn


# Apply the monkey patch immediately
torch.compile = mock_compile


@pytest.fixture(autouse=True, scope="session")
def test_suite_cleanup_thing():
    yield

    build = "./tests/build/"
    print("Cleaning up tests.")

    shutil.rmtree(build)
