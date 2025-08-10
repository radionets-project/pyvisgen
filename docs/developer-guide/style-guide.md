(coding-style)=
# Coding Style Guide

`pyvisgen` follows the [PEP8 style guide][pep8] for Python. This is enforced via the [Ruff][ruff]
linter and code formatter and through the `pre-commit` hook set up in {ref}`pre_commit`.


(docs_style)=
## API Documentation Style

All functions, classes including their methods, and modules should contain API documentation
in their docstrings. We're adhering to the [`numpydoc` style guide][numpydoc], that in it's
simplest format consists of a summary, and a description of parameters and
return values.

:::{code-block} python
:caption: Example of a docstring in `pyvisgen` (taken from {func}`~pyvisgen.utils.config.read_data_set_conf`).

def read_data_set_conf(conf_toml: str | Path) -> dict:
    """Read toml data set configuration file and convert
    it into a dictionary.

    Parameters
    ----------
    conf_toml : str or Path
        Path to config file.

    Returns
    -------
    conf : dict
        Simulation configuration.
    """
:::

Please make sure you include types for each argument if possible.


## Type Hinting

We follow [PEP484][pep484] and [PEP544][pep544] for type hinting. Please make sure you add
type hints to function definitions whenever possible.


## Unit Tests

All code you write should have associated *unit tests*, i.e. persistent tests for a single functionality or
library (e.g. functions, methods, or classes). The tests are used to ensure the code works
and handles exceptions as expected, and to reduce bugs.

`pyvisgen` uses [`pytest`][pytest] as testing framework (see {ref}`testing`). All tests
are located in the `tests` directory in the https://github.com/radionets-project/pyvisgen
main repository. Separate files are used for each module.

When developing new features, make testing part of that development process.
Make sure to write unit tests, that...

1. At least execute all new functionalities
2. Test I/O operations if required, e.g. through simple test inputs
3. Cover edge and exception cases

Additionally, any time you find and fix a bug, it is also best practice
to add a unit test to ensure that specific bug does not appear again.

To measure the test coverage[^1] of the codebase, we use [`Coverage.py`][coveragepy]
with [`pytest-cov`][pytestcov] as plugin for `pytest`. `pytest-cov` can be invoked
using the `--cov` flag:

```shell-session
$ pytest --cov
```

Coverage is also automatically reported using [Codecov][codecov] with our CI: ![codecovbadge](https://codecov.io/github/radionets-project/pyvisgen/badge.svg)

Further, we also use [`pytest-xdist`][pytestxdist] to allow distributing tests
across multiple CPUs to speed up test execution, e.g. using all available CPUs:

```shell-session
$ pytest -n auto
```


[^1]: That is the percentage of code covered by tests. See, e.g., [this article on coverage][atlassian-cov].

[pep8]: https://peps.python.org/pep-0008/
[ruff]: https://docs.astral.sh/ruff/
[numpydoc]: https://numpydoc.readthedocs.io/en/latest/format.html
[pep484]: https://peps.python.org/pep-0484/
[pep544]: https://peps.python.org/pep-0544/
[pytest]: https://docs.pytest.org/en/stable/
[coveragepy]: https://coverage.readthedocs.io/en/7.10.2/
[pytestcov]: https://pytest-cov.readthedocs.io/en/latest/
[codecov]: https://app.codecov.io/github/radionets-project/pyvisgen
[codecovbadge]:  https://codecov.io/github/radionets-project/pyvisgen/badge.svg
[pytestxdist]: https://pytest-xdist.readthedocs.io/en/stable/
[atlassian-cov]: https://www.atlassian.com/continuous-delivery/software-testing/code-coverage
