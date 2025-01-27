.. _getting_started_dev:

******************************
Getting Started for Developers
******************************

We strongly recommend using the `Miniforge3 conda distribution <https://github.com/conda-forge/miniforge>`_
that ships the package installer ``mamba``, a C++ reimplementation of ``conda``.

.. warning::

   The following guide is used only if you want to *develop* the
   ``pyvisgen`` package, if you just want to write code that uses it
   as a dependency, you can install ``pyvisgen`` using pip.
   See :ref:`getting_started_users`


Setting Up the Development Environment
======================================

We provide a conda environment with all packages needed for development of pyvisgen
that can be installed via:

.. code-block:: console

    $ mamba env create -f environment.yml


Next, switch to this new virtual environment:

.. code-block:: console

    $ mamba activate pyvisgen

You will need to run that last command any time you open a new
terminal session to activate the conda environment.


Installing pyvisgen in Development Mode
=======================================

To install pyvisgen inside the ``pyvisgen`` environment
(or any environment for that matter), just run

.. code-block:: console

   $ pip install -e .

This installs the package in editable mode, meaning that you won't have to rerun
the installation for code changes to take effect. For greater changes such as
adding new entry points, the command may have to be run again.
