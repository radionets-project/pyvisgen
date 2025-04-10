:html_theme.sidebar_secondary.remove: true
:html_theme.sidebar_primary.remove: true

.. _pyvisgen:

.. show title in tab name but not on index page
.. raw:: html

   <div style="height: 0; visibility: hidden;">

========
Pyvisgen
========

.. raw:: html

   </div>

.. currentmodule:: pyvisgen

.. image:: _static/pyvisgen.webp
   :class: only-light
   :align: center
   :width: 90%
   :alt: The pyvisgen logo.

.. image:: _static/pyvisgen_dark.webp
   :class: only-dark
   :align: center
   :width: 90%
   :alt: The pyvisgen logo.

|

**Version**: |version| | **Date**: |today|

**Useful links**:
`Source Repository <https://github.com/radionets-project/pyvisgen>`__ |
`Issue Tracker <https://github.com/radionets-project/pyvisgen/issues>`__ |
`Pull Requests <https://github.com/radionets-project/pyvisgen/pulls>`__

**License**: `MIT <https://github.com/radionets-project/pyvisgen/blob/main/LICENSE>`__

**Python**: |python_requires|

`pyvisgen` is a python implementation of the Radio Interferometer Measurement Equation (RIME)
formalism inspired by the VISGEN tool of the `MIT Array Performance Simulator <https://github.com/piyanatk/MAPS>`__
developed at `Haystack Observatory <https://www.haystack.mit.edu/astronomy/>`__. The RIME is used to simulate
the measurement process of a radio interferometer. A gridder is also implemented to process the resulting
visibilities and convert them to images suitable as input for the neural networks developed in the
`radionets repository <https://github.com/radionets-project/radionets>`__.


.. _pyvisgen_docs:

.. toctree::
  :maxdepth: 1
  :hidden:

  user-guide/index
  developer-guide/index
  api-reference/index
  changelog



.. grid:: 1 2 2 3

    .. grid-item-card::

        :octicon:`book;40px`

        User Guide
        ^^^^^^^^^^

        Learn how to get started as a user. This guide
        will help you install pyvisgen.

        +++

        .. button-ref:: user-guide/index
            :expand:
            :color: primary
            :click-parent:

            To the user guide


    .. grid-item-card::

        :octicon:`person-add;40px`

        Developer Guide
        ^^^^^^^^^^^^^^^

        Learn how to get started as a developer.
        This guide will help you install pyvisgen for development
        and explains how to contribute.

        +++

        .. button-ref:: developer-guide/index
            :expand:
            :color: primary
            :click-parent:

            To the developer guide


    .. grid-item-card::

        :octicon:`code;40px`

        API Docs
        ^^^^^^^^

        The API docs contain detailed descriptions of
        of the various modules, classes and functions
        included in pyvisgen.

        +++

        .. button-ref:: api-reference/index
            :expand:
            :color: primary
            :click-parent:

            To the API docs
