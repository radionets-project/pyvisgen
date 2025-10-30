import shutil
from pathlib import Path

import pytest


@pytest.fixture(autouse=True, scope="session")
def test_suite_cleanup_thing():
    yield

    build = Path("./tests/build/")
    print("\nCleaning up tests.")

    if build.is_dir():
        shutil.rmtree(build)

    print(f"Removed {build.resolve().absolute()}")
