(getting_started_dev)=

# Getting Started for Developers

We strongly recommend using the [Miniforge3 conda distribution](https://github.com/conda-forge/miniforge)
that ships the package installer [`mamba`][mamba], a C++ reimplementation of ``conda``.

:::{warning}
The following guide is used only if you want to *develop* the
`pyvisgen` package, if you just want to write code that uses it
as a dependency, you can install `pyvisgen` through one of the
installation methods in {ref}`getting_started_users`
:::


## Setting Up the Development Environment

We provide a [`mamba`][mamba]/`conda` environment with all packages needed for development of pyvisgen
that can be installed via:

```shell-session
$ mamba env create --file=environment-dev.yml
```

Next, activate this new virtual environment:

```shell-session
$ mamba activate pyvisgen
```

You will need to run that last command any time you open a new
terminal session to activate the [`mamba`][mamba]/`conda` environment.


## Installing `pyvisgen` in Development Mode

:::{note}
We recommend using the `uv` package manager to install ``pyvisgen``
and its dependencies. Never heard of `uv`? See [the documentation][uv] for more.
:::

To install `pyvisgen` in your virtual environment, just run

```shell-session
$ uv pip install --group dev -e .
```
in the root of the directory (the directory that contains the `pyproject.toml` file).
This installs the package in editable mode, meaning that you won't have to rerun
the installation for code changes to take effect. For greater changes such as
adding new entry points, the command may have to be run again.

:::{attention}
Make sure you include the `--group` flag to install the `dev` dependency group, which
provides all the necessary dependencies for development on `pyvisgen`.
:::


(pre_commit)=
## Further Setup

We are using [`pre-commit`][pre-commit] with [Ruff][ruff] as linter and formatter for automatic code adherence
to the {ref}`coding-style`. Install the `pre-commit` hooks:
```shell-session
$ pre-commit install
```
The pre-commit hooks will then run every time you commit something. If any of the tools
reports a problem, the commit will be aborted and you will have to fix the issues first.
Usually, a failing `pre-commit` hook indicates code not complying with the style guide.
Once all problems are fixed, you can try committing again, and the changes will be accepted.

To run `pre-commit` manually, call:
```shell-session
$ pre-commit run
```
Or, to run it on all files:
```shell-session
$ pre-commit run --all-files
```
The [Ruff][ruff] hook uses the configuration in [`pyproject.toml`][pyvisgen-pyproject] for linting and formatting.


## Next Steps

Check out {ref}`contributions` and {ref}`coding-style` to learn how to contribute
to `pyvisgen` as a developer.


[mamba]: https://mamba.readthedocs.io/en/latest/
[uv]: https://docs.astral.sh/uv/
[ruff]: https://docs.astral.sh/ruff/
[pyvisgen-pyproject]: https://github.com/radionets-project/pyvisgen/blob/main/pyproject.toml
