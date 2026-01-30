import pytest

from pyvisgen.utils.batch_size import _reduce_batch_size


@pytest.mark.parametrize(
    "batch_size,factor", [(10, 0.5), (100, 0.5), (100, 0.3), (1000, 0.25)]
)
def test_reduce_batch_size(batch_size: int, factor: float) -> None:
    assert _reduce_batch_size(batch_size, factor) == int(batch_size * factor)
