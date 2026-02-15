Quick Start
===========

Basic Usage
-----------

All SLICOT routines expect arrays in **Fortran (column-major) order**.
Use ``order='F'`` when creating NumPy arrays:

.. code-block:: python

   import numpy as np
   from ctrlsys import ab01md

   # Create arrays in Fortran order
   A = np.array([[1, 2], [3, 4]], dtype=float, order='F')
   b = np.array([1, 0], dtype=float, order='F')

   # Call the routine
   result = ab01md('I', A, b, 0.0)

   # Results are returned as a tuple
   A_out, b_out, ncont, Z, tau, info = result
   print(f'Controllable states: {ncont}')
   print(f'Exit code: {info}')

Available Routines
------------------

Routines are organized by prefix:

* **AB** - Analysis and synthesis routines
* **BB/BD** - Benchmark and data generation
* **DE/DF/DG/DK** - Descriptor systems
* **FB/FD** - Factorizations
* **MA/MB/MC/MD** - Matrix operations
* **NF** - Numerical functions
* **SB/SG** - State-space computations
* **TB/TC/TD/TF/TG** - Transformations
* **UD/UE** - Utility routines

Import specific routines:

.. code-block:: python

   from ctrlsys import ab01md, sb03md, mb03rd

Or import everything:

.. code-block:: python

   from ctrlsys import *

Error Handling
--------------

Most routines return an ``info`` value:

* ``info = 0``: Success
* ``info < 0``: Invalid argument (argument number is ``-info``)
* ``info > 0``: Routine-specific warning or error

.. code-block:: python

   result = ab01md('I', A, b, 0.0)
   info = result[-1]  # Last element is always info

   if info != 0:
       print(f'Warning/Error: info = {info}')
