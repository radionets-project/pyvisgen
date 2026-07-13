import pytest

from pyvisgen.exceptions import OptionalDependencyMissing


def test_optional_dependency_missing_exception():
    opt_dep = "plot"

    with pytest.raises(
        OptionalDependencyMissing,
        match=rf"you need to install pyvisgen with the \[{opt_dep}\]",
    ):
        raise OptionalDependencyMissing(opt_dep)
