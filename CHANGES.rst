Pyvisgen v0.2.0 (2024-06-12)
============================


API Changes
-----------


Bug Fixes
---------

- fix baseline num calculation
- fix wavelength scaling
- fix lm grid calculation
- fix gridding so that it fits the numpy fft gridding [`#28 <https://github.com/radionets-project/pyvisgen/pull/28>`__]


New Features
------------

- implement GPU support for visibility calculations
- new grid mode:

  - when more than one visibility falls into the same pixel, only the first is calculated
  - define grid before calculation

- new dense mode:

  - calculate visibilities for a dense uv grid
  - simulate ideal interferometer response

- add sensitivity cut in image space:

  - avoid calculation of pixel values below detection threshold
  - significantly speed-up simulations

- add torch compile to RIME functions [`#28 <https://github.com/radionets-project/pyvisgen/pull/28>`__]


Maintenance
-----------

- delete unused code and relicts
- change from numpy arrays to torch tensors
- change some of the keywords to more common phrases inside the toml config
- update default data_set.toml
- delete old config examples
- avoid torch einsum for better readability of the code [`#28 <https://github.com/radionets-project/pyvisgen/pull/28>`__]


Refactoring and Optimization
----------------------------

- refactor data classes (Visibilities, Baselines)
- add observation class, which holds all relevant information
- drop scan-wise splitting in visibilities calculations, but split all valid baselines equally
- refactor RIME components (currently only uncorrupted available) [`#28 <https://github.com/radionets-project/pyvisgen/pull/28>`__]
- refactor baseline calculations by replacing loops with pytorch built-in methods


Pyvisgen v0.1.4 (2023-11-09)
============================


API Changes
-----------


Bug Fixes
---------

- fix shape of `num_ifs`

  - delete additional bin in masking
  - fix ra dec bug [`#25 <https://github.com/radionets-project/pyvisgen/pull/25>`__]


New Features
------------

- update ci:

  - change conda to mamba
  - install towncrier [`#24 <https://github.com/radionets-project/pyvisgen/pull/24>`__]


Maintenance
-----------

- update readme [`#26 <https://github.com/radionets-project/pyvisgen/pull/26>`__]
- add docstrings

  - delete unused files [`#27 <https://github.com/radionets-project/pyvisgen/pull/27>`__]


Refactoring and Optimization
----------------------------
