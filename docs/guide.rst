Task-Oriented Guide
====================

This guide maps common control engineering tasks to the appropriate SLICOT routines.
All functions are available directly from the ``slicot`` module:

.. code-block:: python

   from slicot import sb02md, ab09ad

The guide is organized in two tiers. **Common workflows** covers the tasks that
most control engineers encounter — LQR design, model reduction, system norms —
matching the routines used by libraries like `python-control <https://python-control.org>`_.
**Equation solvers and specialized** provides direct access to the underlying numerical
routines for advanced users.

.. contents:: Tasks
   :local:
   :depth: 1


.. rubric:: Common Workflows

LQR and LQG Design
-------------------

The most common control design task: solve an algebraic Riccati equation, then compute
a state-feedback gain. Continuous LQR uses :func:`~slicot.sb02md` (CARE), discrete LQR
uses :func:`~slicot.sg02ad` (DARE). For LQG, also compute Gramians via :func:`~slicot.sb03md`.

- :func:`~slicot.sb02md` -- Continuous-time ARE (standard) — use for ``care()`` / ``lqr()``
- :func:`~slicot.sb02mt` -- Construct the Riccati equation data from Q, R, S weight matrices
- :func:`~slicot.sg02ad` -- Generalized ARE (continuous or discrete) — use for ``dare()`` / ``dlqr()``
- :func:`~slicot.sb02od` -- Continuous-time ARE (generalized, with weighting)
- :func:`~slicot.sb02rd` -- Continuous-time ARE (Schur method, condition estimate)


Pole Placement
--------------

Place closed-loop eigenvalues at desired locations via state feedback.

- :func:`~slicot.sb01bd` -- Pole assignment via state feedback (Varga algorithm — robust, handles repeated poles)
- :func:`~slicot.sb01md` -- Pole assignment for multi-input systems (Ackermann method)


System Norms
------------

Compute norms of transfer functions for performance and robustness analysis.

- :func:`~slicot.ab13dd` -- H-infinity / L-infinity norm of a transfer function
- :func:`~slicot.ab13bd` -- H2 norm (L2 norm)
- :func:`~slicot.ab13md` -- Structured singular value (mu) — used for disk margin analysis
- :func:`~slicot.ab13ed` -- Frequency response peak
- :func:`~slicot.ab13fd` -- L-infinity norm of a transfer matrix
- :func:`~slicot.ab13hd` -- Hankel norm


Model Reduction
---------------

Reduce model order while preserving key input-output behavior.

**Stable systems:**

- :func:`~slicot.ab09ad` -- Balanced truncation (BT)
- :func:`~slicot.ab09bd` -- Square-root balanced truncation (SRBT)
- :func:`~slicot.ab09cd` -- Balanced stochastic truncation (BST)
- :func:`~slicot.ab09md` -- Hankel-norm approximation
- :func:`~slicot.ab09nd` -- Balanced reduction with frequency weighting — used by ``balred()`` with ``matchdc``

**Unstable systems (additive decomposition):**

- :func:`~slicot.ab09dd` -- BT with coprime factorization
- :func:`~slicot.ab09ed` -- Hankel-norm via coprime factorization

**Frequency-weighted reduction:**

- :func:`~slicot.ab09fd` -- Frequency-weighted BT
- :func:`~slicot.ab09gd` -- Frequency-weighted BT (stability-preserving)
- :func:`~slicot.ab09hd` -- Frequency-weighted Hankel-norm
- :func:`~slicot.ab09id` -- Frequency-weighted BT (general weighting)
- :func:`~slicot.ab09jd` -- Frequency-weighted BT (general, stability-preserving)
- :func:`~slicot.ab09kd` -- Frequency-weighted Hankel-norm (general weighting)


H-infinity and H2 Control
--------------------------

Design controllers for robust performance under uncertainty.

- :func:`~slicot.sb10ad` -- H-infinity controller (continuous, general)
- :func:`~slicot.sb10dd` -- H-infinity controller (discrete)
- :func:`~slicot.sb10hd` -- H2 optimal controller (discrete) — used by ``h2syn()``
- :func:`~slicot.sb10fd` -- H-infinity controller (continuous, simplified)
- :func:`~slicot.sb10ed` -- H-infinity controller (loopshaping)
- :func:`~slicot.sb10id` -- H-infinity controller (LMI-based, continuous)
- :func:`~slicot.sb10jd` -- H-infinity controller (LMI-based, discrete)
- :func:`~slicot.sb10ld` -- H-infinity controller (coprime factorization)


Controllability, Observability, and Gramians
--------------------------------------------

Structural properties of state-space systems and controllability/observability Gramians.

**Gramians:**

- :func:`~slicot.sb03md` -- Controllability/observability Gramians via Lyapunov equations
- :func:`~slicot.sb03od` -- Cholesky-factored Gramians via Lyapunov equations
- :func:`~slicot.sg03ad` -- Gramians for descriptor systems (generalized Lyapunov)

**Structural decompositions:**

- :func:`~slicot.ab01md` -- Controllability staircase form (upper)
- :func:`~slicot.ab01nd` -- Controllability staircase form (lower)
- :func:`~slicot.ab01od` -- Controllable/uncontrollable decomposition
- :func:`~slicot.ab08nd` -- Transmission zeros and normal rank
- :func:`~slicot.ab08md` -- Right Kronecker indices and transmission zeros
- :func:`~slicot.ag08bd` -- Transmission zeros for descriptor systems


Frequency and Time Response
---------------------------

Evaluate system responses in frequency and time domains.

**Frequency response:**

- :func:`~slicot.tb05ad` -- Frequency response at given complex frequency points

**Time response:**

- :func:`~slicot.tf01md` -- Output response of a state-space system to a given input sequence
- :func:`~slicot.tf01nd` -- Output response via Hessenberg form


Transfer Function Conversion
-----------------------------

Convert between state-space and transfer function representations.

- :func:`~slicot.tb04ad` -- State-space to transfer function (SISO/MIMO)
- :func:`~slicot.tb03ad` -- Transfer function to state-space (polynomial form)
- :func:`~slicot.td04ad` -- Transfer function to descriptor state-space
- :func:`~slicot.tc04ad` -- Left MFD to right MFD conversion


System Interconnections
-----------------------

Connect state-space systems in series, parallel, and feedback configurations.

- :func:`~slicot.ab05md` -- Series (cascade) connection G2 * G1
- :func:`~slicot.ab05nd` -- Feedback connection
- :func:`~slicot.ab05od` -- Parallel connection G1 + G2
- :func:`~slicot.ab05pd` -- Series connection (with scaling)
- :func:`~slicot.ab05qd` -- Append (block diagonal) two systems
- :func:`~slicot.ab05rd` -- Feedback connection (with scaling)
- :func:`~slicot.ab05sd` -- Series-parallel connection


State-Space Transformations
----------------------------

Transform, balance, and convert system representations.

- :func:`~slicot.tb01pd` -- Minimal realization (controllable + observable) — used by ``minreal()``
- :func:`~slicot.tb01wd` -- State-space balancing
- :func:`~slicot.tb01td` -- Upper block-triangular form (similarity)
- :func:`~slicot.mb03rd` -- Real Schur form / modal decomposition — used by ``modal_form()``
- :func:`~slicot.ab07md` -- Inverse of a state-space system
- :func:`~slicot.ab07nd` -- Dual (transpose) of a state-space system


Continuous-Discrete Conversion
------------------------------

Convert between continuous-time and discrete-time representations.

- :func:`~slicot.ab04md` -- Bilinear transformation (c2d / d2c)


.. rubric:: Equation Solvers and Specialized

Riccati Equations
-----------------

Solve algebraic Riccati equations (ARE) arising in LQR/LQG and H-infinity control.
See also `LQR and LQG Design`_ above for the typical workflow.

- :func:`~slicot.sb02md` -- Continuous-time ARE (standard)
- :func:`~slicot.sb02od` -- Continuous-time ARE (generalized, with weighting)
- :func:`~slicot.sb02rd` -- Continuous-time ARE (Schur method, condition estimate)
- :func:`~slicot.sg02ad` -- Generalized ARE for descriptor systems (continuous or discrete)


Lyapunov and Stein Equations
----------------------------

Solve Lyapunov (continuous) and Stein/discrete-Lyapunov equations for stability analysis, Gramians, and covariance computation.
See also `Controllability, Observability, and Gramians`_ above for Gramian workflows.

- :func:`~slicot.sb03md` -- Continuous or discrete Lyapunov equation (standard)
- :func:`~slicot.sb03od` -- Continuous or discrete Lyapunov equation (Cholesky factor)
- :func:`~slicot.sg03ad` -- Generalized Lyapunov equation for descriptor systems
- :func:`~slicot.sg03bd` -- Generalized Lyapunov equation (Cholesky factor)


Sylvester Equations
-------------------

Solve Sylvester and generalized Sylvester matrix equations.

- :func:`~slicot.sb04md` -- Continuous Sylvester equation AX + XB = C
- :func:`~slicot.sb04nd` -- Discrete Sylvester equation AXB + X = C
- :func:`~slicot.sb04qd` -- Continuous Sylvester equation (Hessenberg-Schur method)
- :func:`~slicot.sb04rd` -- Discrete Sylvester equation (Hessenberg-Schur method)


Spectral Factorization
-----------------------

Compute spectral factors for optimal and robust control.

- :func:`~slicot.sb08cd` -- Left spectral factorization (continuous)
- :func:`~slicot.sb08dd` -- Right spectral factorization (continuous)
- :func:`~slicot.sb08ed` -- Left spectral factorization (discrete)
- :func:`~slicot.sb08fd` -- Right spectral factorization (discrete)
- :func:`~slicot.sb08gd` -- Left coprime factorization (continuous)
- :func:`~slicot.sb08hd` -- Right coprime factorization (continuous)


System Identification
---------------------

Estimate state-space models from measured input/output data.

- :func:`~slicot.ib01ad` -- Subspace identification: compute system order
- :func:`~slicot.ib01bd` -- Subspace identification: estimate A, B, C, D
- :func:`~slicot.ib01cd` -- Subspace identification: estimate covariance matrices
- :func:`~slicot.ib03ad` -- Wiener system identification (nonlinear)


Kalman Filtering
----------------

Sequential (square-root) filtering and smoothing.

- :func:`~slicot.fb01qd` -- Square-root Kalman filter (one step)
- :func:`~slicot.fb01rd` -- Square-root Kalman filter (information form)
- :func:`~slicot.fb01sd` -- Square-root covariance filter
- :func:`~slicot.fb01vd` -- Kalman filter (conventional form)


Matrix Exponential
------------------

Compute matrix exponentials and related quantities.

- :func:`~slicot.mb05md` -- Matrix exponential via Pade approximation
- :func:`~slicot.mb05nd` -- Matrix exponential and integral
- :func:`~slicot.mb05od` -- Matrix exponential of a Hamiltonian matrix


Polynomial Operations
---------------------

Operations on real and complex polynomials.

- :func:`~slicot.mc01md` -- Evaluate polynomial at a point
- :func:`~slicot.mc01pd` -- Greatest common divisor of two polynomials
- :func:`~slicot.mc01qd` -- Roots to polynomial coefficients
- :func:`~slicot.mc01rd` -- Polynomial coefficients to roots (companion)


Descriptor System Operations
-----------------------------

Operations specific to descriptor (generalized) state-space systems E*dx/dt = A*x + B*u.

- :func:`~slicot.tg01fd` -- Reduce descriptor system to SVD-like form
- :func:`~slicot.tg01hd` -- Reduce descriptor system to Hessenberg-triangular form
- :func:`~slicot.tg01jd` -- Reduce descriptor system to block diagonal form
- :func:`~slicot.ag07bd` -- Inverse of a descriptor system
- :func:`~slicot.ag08bd` -- Transmission zeros of a descriptor system


Benchmark Systems
-----------------

Generate standard benchmark state-space models for testing.

- :func:`~slicot.bb01ad` -- Continuous-time benchmark examples
- :func:`~slicot.bb02ad` -- More continuous-time benchmark examples
- :func:`~slicot.bb03ad` -- Discrete-time benchmark examples
- :func:`~slicot.bb04ad` -- Parametric benchmark examples
