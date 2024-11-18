=========================
pyvisgen |ci| |codecov|
=========================

.. |ci| image:: https://github.com/radionets-project/pyvisgen/workflows/CI/badge.svg?branch=main
    :target: https://github.com/radionets-project/pyvisgen/actions/workflows/ci.yml?branch=main
    :alt: Test Status

.. |codecov| image:: https://codecov.io/github/radionets-project/pyvisgen/badge.svg
    :target: https://codecov.io/github/radionets-project/pyvisgen
    :alt: Code coverage


Python implementation of the VISGEN tool developed at `Haystack Observatory <https://www.haystack.mit.edu/astronomy/>`_.
It uses the Radio Interferometer Measurement Equation (RIME) to simulate the measurement process of a radio interferometer.
A gridder is also implemented to process the resulting visibilities and convert them to images suitable as input for
the neural networks developed in the `radionets repository <https://github.com/radionets-project/radionets>`_.

Installation
============

You can install the necessary packages in a conda environment of your choice by executing

.. code::

  $ pip install -e .


Usage
=====

There are 3 possible modes at the moment:  ``simulate`` (default), ``slurm``, and ``gridding``. ``simulate`` and ``slurm`` both
utilize the RIME formalism for creating visibilities data. With the option ``gridding``, these visibilities get gridded and prepared
as input images for training a neural network from the radionets framework. The necessary options and variables are set with a ``toml``
file. An exemplary file can be found in ``config/data_set.toml``.

..code::

  $ pyvisgen_create_dataset --mode=simulate some_file.toml


In the examples directory, you can find introductory jupyter notebooks which can be used as an entry point.

Input images
============

As input images for the RIME formalism, we use GAN-generated radio galaxies created by `Rustige et. al. <https://doi.org/10.1093/rasti/rzad016>`_
and `Kummer et. al. <https://doi.org/10.18420/inf2022_38>`_. Below, you can see four example images consisting of FRI and FRII sources.

.. image:: https://github.com/radionets-project/pyvisgen/assets/23259659/285e36f6-74e7-45f1-9976-896a38217880
   :alt: sources

Any image can be used as input for the formalism, as long as they are stored in the h5 format, generated with |h5py|_.

.. |h5py| replace:: ``h5py``
.. _h5py: https://www.h5py.org/

RIME
====

Currently, we use the following expression for the simulation process:

.. math::

  \mathbf{V}_{\mathrm{pq}}(l, m) = \sum\_{l, m} \mathbf{E}\_{\mathrm{p}}(l, m) \mathbf{K}\_{\mathrm{p}}(l, m) \mathbf{B}(l, m) \mathbf{K}^{H}\_{\mathrm{q}}(l, m) \mathbf{E}^{H}\_{\mathrm{q}}(l, m)

Here, :math:`\mathbf{B}(l, m)` corresponds to the source distribution, :math:`\mathbf{K}(l, m) = \exp(-2\pi\cdot i\cdot (ul + vm))` represents
the phase delay, and :math:`\mathbf{E}(l, m) = \mathrm{jinc}\left(\frac{2\pi}{\lambda}d\cdot \theta\_{lm}\right)` the telescope properties,
with :math:`\mathrm{jinc(x)}=\frac{J_1(x}{x}` and :math:`J_1(x)` as the first Bessel function. An exemplary result can be found below.

.. image:: https://github.com/radionets-project/pyvisgen/assets/23259659/858a5d4b-893a-4216-8d33-41d33981354c
   :alt: visibilities

Visualization of Jones matrices
===============================

In this section, you can see visualizations of the matrices :math:`\mathbf{E}(l, m)`  and :math:`\mathbf{K}(l, m)`.

Visualization of the E matrix
-----------------------------
.. image:: https://github.com/radionets-project/pyvisgen/assets/23259659/194a321b-77cd-423b-9d01-c18c0741d6c5
   :alt: visualize_E

Visualization of the K matrix
-----------------------------
.. image:: https://github.com/radionets-project/pyvisgen/assets/23259659/501f487a-498b-4143-b54a-eb0e2f28e417
   :alt: visualize_K
