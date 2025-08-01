(getting_started_dev)=

# Getting Started for Developers

We strongly recommend using the [Miniforge3 conda distribution](https://github.com/conda-forge/miniforge)
that ships the package installer [`mamba`][mamba], a C++ reimplementation of ``conda``.

:::{warning}
The following guide is used only if you want to *develop* the
``pyvisgen`` package, if you just want to write code that uses it
as a dependency, you can install ``pyvisgen`` through one of the
installation methods in {ref}`getting_started_users`
:::


## Setting Up the Development Environment

We provide a [`mamba`][mamba]/`conda` environment with all packages needed for development of pyvisgen
that can be installed via:

```shell-shell
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

To install pyvisgen in your virtual environment, just run

```shell-shell
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

[mamba]: https://mamba.readthedocs.io/en/latest/
[uv]: https://docs.astral.sh/uv/
