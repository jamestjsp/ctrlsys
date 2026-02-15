SLICOT - Subroutine Library in Control Theory
=============================================

A C11 translation of the SLICOT library with Python bindings.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   guide
   api/index

Installation
------------

.. code-block:: bash

   pip install ctrlsys

Pre-built wheels are available for Linux, macOS, and Windows (Python 3.11-3.13).
OpenBLAS is bundled - no external BLAS/LAPACK installation required.

Quick Example
-------------

.. code-block:: python

   import numpy as np
   from ctrlsys import ab01md

   A = np.array([[1, 2], [3, 4]], dtype=float, order='F')
   b = np.array([1, 0], dtype=float, order='F')
   result = ab01md('I', A, b, 0.0)
   print('Controllable states:', result[2])

Links
-----

* `PyPI <https://pypi.org/project/ctrlsys/>`_
* `GitHub <https://github.com/jamestjsp/slicot>`_
* `Original SLICOT <http://slicot.org/>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
