Pyvisgen v0.3.0 (2025-07-02)
============================


API Changes
-----------


Bug Fixes
---------

- Fix shape of `num_ifs`

  - Delete additional bin in masking
  - Fix ra dec bug [`#25 <https://github.com/radionets-project/pyvisgen/pull/25>`__]

- Fix baseline num calculation

  - Fix wavelength scaling
  - Fix lm grid calculation
  - Fix gridding so that it fits the numpy fft gridding [`#28 <https://github.com/radionets-project/pyvisgen/pull/28>`__]

- Fix a numerical issue in the lm grid calculation, caused by adding a big number to small values in the lm grid

  - Use torch.float64 for rd grid and lm grid calculation [`#32 <https://github.com/radionets-project/pyvisgen/pull/32>`__]

- Fix gridding in `pyvisgen.simulation.observation.Observation` methods `create_rd_grid` and `create_lm_grid`
    methods resulting in rotated images

  - Fix `pyvisgen.simulation.observation.ValidBaselineSubset` dataclass field order
  - Fix tests failing because of api change [`#39 <https://github.com/radionets-project/pyvisgen/pull/39>`__]

- Fix image rotation caused by bug in rd/lm grid computation in ``pyvisgen.simulation.observation.Obseravtion``
  - Fix field order in ``pyvisgen.simulation.observation.ValidBaselineSubset`` data class
  - Flip input image at the beginning of ``pyvisgen.simulation.visibility.vis_loop`` to ensure correct indexing, e.g. for plotting [`#40 <https://github.com/radionets-project/pyvisgen/pull/40>`__]

- Fixed random number drawing in tests by changing the location of the seed override [`#44 <https://github.com/radionets-project/pyvisgen/pull/44>`__]

- Update the order of simulated bandwidths in the fits writer to the standard found from converted MeerKat observations

  - Tried to fix polarisation infos antenna hdu [`#49 <https://github.com/radionets-project/pyvisgen/pull/49>`__]

- Fix bug in feed rotation/parallactic angle computation in RIME [`#57 <https://github.com/radionets-project/pyvisgen/pull/57>`__]

- Fix observation dec not on same device as r [`#62 <https://github.com/radionets-project/pyvisgen/pull/62>`__]

- `examples/ideal_interferometer.ipynb`: Added a new code cell to create an `lm_grid` with used fov parameters.

  - Fixed missing images for the `lm_grid` in the docs. [`#63 <https://github.com/radionets-project/pyvisgen/pull/63>`__]

- Add quick fix of the annoying import warning when using `tqdm.autonotbook`: Use `tqdm.auto` instead. This does not create any warnings, as written in the `tqdm` documentation (https://tqdm.github.io/docs/shortcuts/#tqdmauto). [`#65 <https://github.com/radionets-project/pyvisgen/pull/65>`__]


New Features
------------

- Implement GPU support for visibility calculations

  - New grid mode:

    - When more than one visibility falls into the same pixel, only the first is calculated
    - Define grid before calculation

  - New dense mode:

    - Calculate visibilities for a dense uv grid
    - Simulate ideal interferometer response

  - Add sensitivity cut in image space:

    - Avoid calculation of pixel values below detection threshold
    - Significantly speed-up simulations

  - Add torch compile to RIME functions [`#28 <https://github.com/radionets-project/pyvisgen/pull/28>`__]

- Changes to `vis_loop` function in `visibility.py`:

  - Add a an optional tqdm progress bar to get a visual confirmation the calculation is still running
  - Add optional `batch_size` parameter to control memory consumption [`#33 <https://github.com/radionets-project/pyvisgen/pull/33>`__]

- Add class `Polarisation` to `pyvisgen.simulation.visibility` that is called in `vis_loop`

    - Added linear, circular, and no polarisation options

  - Update `pyvisgen.simulation.visibility.Visibilities` dataclass to also store polarisation degree tensors
  - Add keyword arguments for polarisation simulation to `pyvisgen.simulation.observation.Observation` class
  - Add parallactic angle computation [`#39 <https://github.com/radionets-project/pyvisgen/pull/39>`__]

- ``pyvisgen.layouts.get_array_layout`` now also accepts custom layouts stored in a ``pd.DataFrame`` [`#46 <https://github.com/radionets-project/pyvisgen/pull/46>`__]

- Add docs [`#47 <https://github.com/radionets-project/pyvisgen/pull/47>`__]

- Added optional auto scaling for batchsize in vis_loop [`#48 <https://github.com/radionets-project/pyvisgen/pull/48>`__]

- Add new gridder that can handle vis data returned by the ``vis_loop`` [`#53 <https://github.com/radionets-project/pyvisgen/pull/53>`__]

- Add ideal interferometer simulation guide to documentation (in `User Guide`)

  - Add example notebook `ideal_interferometer.ipynb` containing full code for the user guide entry
  - Change primary and primary highlight colors for light theme in `_static/pyvisgen.css` to darker greens to be more visible
  - Change maintainers in `pyproject.toml` [`#58 <https://github.com/radionets-project/pyvisgen/pull/58>`__]

- - Add DSA-2000 layouts [`#61 <https://github.com/radionets-project/pyvisgen/pull/61>`__]

- - Add new quickstart CLI tool that creates a copy of the default configuration at the specified path [`#73 <https://github.com/radionets-project/pyvisgen/pull/73>`__]


Maintenance
-----------

- Update readme [`#26 <https://github.com/radionets-project/pyvisgen/pull/26>`__]

- Add docstrings
  - Delete unused files [`#27 <https://github.com/radionets-project/pyvisgen/pull/27>`__]

- Delete unused code and relicts

  - Change from numpy arrays to torch tensors
  - Change some of the keywords to more common phrases inside the toml config
  - Update default data_set.toml
  - Delete old config examples
  - Avoid torch einsum for better readability of the code
  - Update `ci.yml` and `workflow.yml` for node20 [`#28 <https://github.com/radionets-project/pyvisgen/pull/28>`__]

- Add missing changelog [`#29 <https://github.com/radionets-project/pyvisgen/pull/29>`__]

- Use observation class to pass sampling options to the fits writer

  - Include writer in tests [`#31 <https://github.com/radionets-project/pyvisgen/pull/31>`__]

- Use c from scipy in scan.py [`#32 <https://github.com/radionets-project/pyvisgen/pull/32>`__]

- Switch from setup.py to pyproject.toml [`#35 <https://github.com/radionets-project/pyvisgen/pull/35>`__]

- Fix package name and url in pyproject.toml

  - Remove obsolete setup.py
  - Fix setuptools find packages path in pyproject.toml
  - Fix formatting of pyproject.toml [`#36 <https://github.com/radionets-project/pyvisgen/pull/36>`__]

- Create new dev environment file that contains pytorch-gpu and pytorch-cuda [`#37 <https://github.com/radionets-project/pyvisgen/pull/37>`__]

- Change pyvisgen.simulation.visibility.Visibilities dataclass component names from stokes components (I , Q, U, and V)
    to visibilities constructed from the stokes components (`V_11`, `V_22`, `V_12`, `V_21`)

  - Change indices for stokes components according to AIPS Memo 114

    - Indices will be set automatically depending on simulated polarisation

  - Update comment strings in FITS files
  - Update docstrings accordingly in `pyvisgen.simulation.visibility.vis_loop` and `pyvisgen.simulation.observation.Observation` [`#39 <https://github.com/radionets-project/pyvisgen/pull/39>`__]

- Switch README to reStructuredText

  - Add Codecov badge [`#45 <https://github.com/radionets-project/pyvisgen/pull/45>`__]

- Drop integration time in fits writer (also missing fits files which are converted from ms files)

  - Update saving of visibility dates to modern standards
  - Use infos from observation class [`#49 <https://github.com/radionets-project/pyvisgen/pull/49>`__]

- Increase verbosity of tests in CI [`#50 <https://github.com/radionets-project/pyvisgen/pull/50>`__]

- Complete rewrite of dataset creation routine ``pyvisgen.simulation.data_set.SimulateDataSet``

    - Accessible using a classmethod to load a config file
    - Add optional multithreading support
    - Draw and fully test parameters before simulation loop. Previously this was done in the loop and tests were only performed for two time steps
    - Support for polarization

  - Add new default config file for new dataset creation routine
  - Update CLI tool for dataset creation routine
  - Allow passing HDF5 key in ``pyvisgen.utils.data.open_bundles``
  - Restructure ``pyvisgen.gridding`` module by adding a ``utils`` submodule that contains all utility functions that previously were in the ``gridder`` submodule

    - Also fix parts of the utility functions

  - Update and fix tests [`#53 <https://github.com/radionets-project/pyvisgen/pull/53>`__]

- Add/update docstrings throughout the codebase [`#54 <https://github.com/radionets-project/pyvisgen/pull/54>`__]

- Remove ``torch.flip`` call in ``visibility.py``

  - Change dense UV grid creation to use ``numpy.float128`` and convert to ``torch.float64`` afterwards to fix numerical instabilities
  - Change integration in ``scan.py`` to return ``int_f`` instead of ``int_t``, removed time integration
  - Exclude dense calculations from code coverage due to lack of GPU computations in GitHub actions [`#56 <https://github.com/radionets-project/pyvisgen/pull/56>`__]

- Fix docs index and readme text [`#60 <https://github.com/radionets-project/pyvisgen/pull/60>`__]

- Add linting CI job

  - Fix attribute error in ``pyvisgen.simulation`` [`#67 <https://github.com/radionets-project/pyvisgen/pull/67>`__]


Refactoring and Optimization
----------------------------

- Refactor data classes (Visibilities, Baselines)

  - Add observation class, which holds all relevant information
  - Drop scan-wise splitting in visibilities calculations, but split all valid baselines equally
  - Refactor RIME components (currently only uncorrupted available)
  - Refactor baseline calculations by replacing loops with pytorch built-in methods [`#28 <https://github.com/radionets-project/pyvisgen/pull/28>`__]

- Improve hour angle calculation via array-wise operations [`#30 <https://github.com/radionets-project/pyvisgen/pull/30>`__]

- Use `obs.layout` instead of passing the layout name separately in `pyvisgen.fits.writer.create_vis_hdu` [`#38 <https://github.com/radionets-project/pyvisgen/pull/38>`__]

- Added optional `normalize` parameter to `pyvisgen.visibility.vis_loop` to decide whether to apply a normalization multiplier of `0.5` (default: True) [`#43 <https://github.com/radionets-project/pyvisgen/pull/43>`__]

- Remove reading of layout files relative to :mod:`pyvisgen.layouts.layout`

  - Move layout files to external resources directory that is shipped with
    the distribution
  - Ship default config with distribution [`#73 <https://github.com/radionets-project/pyvisgen/pull/73>`__]

Pyvisgen v0.2.0 (2024-06-12)
============================


API Changes
-----------


Bug Fixes
---------

- Fix baseline num calculation
- Fix wavelength scaling
- Fix lm grid calculation
- Fix gridding so that it fits the numpy fft gridding [`#28 <https://github.com/radionets-project/pyvisgen/pull/28>`__]


New Features
------------

- Implement GPU support for visibility calculations

- New grid mode:

  - When more than one visibility falls into the same pixel, only the first is calculated
  - Define grid before calculation

- New dense mode:

  - Calculate visibilities for a dense uv grid
  - Simulate ideal interferometer response

- Add sensitivity cut in image space:

  - Avoid calculation of pixel values below detection threshold
  - Significantly speed-up simulations

- Add torch compile to RIME functions [`#28 <https://github.com/radionets-project/pyvisgen/pull/28>`__]


Maintenance
-----------

- Delete unused code and relicts
- Change from numpy arrays to torch tensors
- Change some of the keywords to more common phrases inside the toml config
- Update default data_set.toml
- Delete old config examples
- Avoid torch einsum for better readability of the code [`#28 <https://github.com/radionets-project/pyvisgen/pull/28>`__]


Refactoring and Optimization
----------------------------

- Refactor data classes (Visibilities, Baselines)
- Add observation class, which holds all relevant information
- Drop scan-wise splitting in visibilities calculations, but split all valid baselines equally
- Refactor RIME components (currently only uncorrupted available) [`#28 <https://github.com/radionets-project/pyvisgen/pull/28>`__]
- Refactor baseline calculations by replacing loops with pytorch built-in methods


Pyvisgen v0.1.4 (2023-11-09)
============================


API Changes
-----------


Bug Fixes
---------

- Fix shape of `num_ifs`

  - Delete additional bin in masking
  - Fix ra dec bug [`#25 <https://github.com/radionets-project/pyvisgen/pull/25>`__]


New Features
------------

- Update ci:

  - Change conda to mamba
  - Install towncrier [`#24 <https://github.com/radionets-project/pyvisgen/pull/24>`__]


Maintenance
-----------

- Update readme [`#26 <https://github.com/radionets-project/pyvisgen/pull/26>`__]
- Add docstrings

  - Delete unused files [`#27 <https://github.com/radionets-project/pyvisgen/pull/27>`__]


Refactoring and Optimization
----------------------------
