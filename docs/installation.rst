Installation
============

From PyPI (Recommended)
-----------------------

.. code-block:: bash

   pip install ctrlsys

Pre-built wheels are available for:

* **Linux**: x86_64 (manylinux2014)
* **macOS**: ARM64 (Apple Silicon)
* **Windows**: x86_64

Supported Python versions: 3.11, 3.12, 3.13

OpenBLAS is bundled in the wheels - no external BLAS/LAPACK installation required.

From Source
-----------

Requirements:

* C11 compiler (GCC, Clang, MSVC)
* BLAS/LAPACK development libraries
* Meson build system
* NumPy >= 2.0

.. code-block:: bash

   # Install build dependencies
   pip install meson-python meson numpy

   # Clone and install
   git clone https://github.com/jamestjsp/slicot.git
   cd slicot
   pip install .

Development Installation
------------------------

.. code-block:: bash

   git clone https://github.com/jamestjsp/slicot.git
   cd slicot
   pip install -e ".[test]"
   pytest tests/python/
