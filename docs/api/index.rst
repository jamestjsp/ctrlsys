API Reference
=============

All routines are available from the ``slicot`` module:

.. code-block:: python

   from slicot import ab01md, sb03md

Routine Categories
------------------

Analysis (AB, AG)
~~~~~~~~~~~~~~~~~

Routines for system analysis, controllability, observability.

* ``ab01md`` - Controllable realization for single-input systems
* ``ab01nd`` - Controllable realization for multi-input systems
* ``ab01od`` - Staircase form for controllability
* ``ab04md`` - Discrete-time to continuous-time conversion
* ``ab05md`` - Series connection of systems
* ``ab05nd`` - Parallel connection of systems
* ``ab07md`` - Inverse of a state-space system
* ``ab08md`` - Construction of a regular pencil
* ``ab09ad`` - Balanced truncation approximation

State-Space (SB, SG)
~~~~~~~~~~~~~~~~~~~~

* ``sb01bd`` - Pole assignment for state feedback
* ``sb02md`` - Algebraic Riccati equation (continuous)
* ``sb02od`` - Algebraic Riccati equation (discrete)
* ``sb03md`` - Lyapunov equation solver
* ``sb04md`` - Sylvester equation solver
* ``sb10ad`` - H-infinity controller synthesis

Matrix Operations (MB, MC, MD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``mb01rd`` - Matrix scaling
* ``mb02md`` - QR factorization with pivoting
* ``mb03rd`` - Real Schur form
* ``mb04md`` - Balance a matrix pair

Transformations (TB, TD, TF, TG)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``tb01id`` - State-space to transfer function
* ``tb01pd`` - Minimal realization
* ``td04ad`` - Transfer function to state-space
* ``tg01ad`` - Descriptor system balancing

For complete documentation of each routine, use Python's built-in help:

.. code-block:: python

   from slicot import ab01md
   help(ab01md)
