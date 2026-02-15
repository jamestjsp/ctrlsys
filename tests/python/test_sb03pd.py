"""
Tests for SB03PD: Discrete-time Lyapunov equation solver with separation estimation.

Solves: op(A)' * X * op(A) - X = scale * C
where op(A) = A or A' and C is symmetric.

Also computes separation: sepd(op(A), op(A)') = min norm(op(A)'*X*op(A) - X)/norm(X)
"""

import numpy as np
import pytest


def test_sb03pd_basic_solution():
    """
    Test basic discrete Lyapunov solution with FACT='N'.

    Creates a stable discrete-time system (eigenvalues inside unit circle)
    and validates the Lyapunov equation residual.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sb03pd

    np.random.seed(42)
    n = 3

    # Create a stable discrete-time matrix (eigenvalues inside unit circle)
    a = np.array([
        [0.5, 0.1, 0.0],
        [0.0, 0.3, 0.2],
        [0.0, 0.0, 0.4]
    ], order='F', dtype=float)

    # Symmetric positive definite C
    c = np.array([
        [1.0, 0.2, 0.1],
        [0.2, 1.5, 0.3],
        [0.1, 0.3, 2.0]
    ], order='F', dtype=float)

    # Solve: A' * X * A - X = scale * C
    a_out, u, x, scale, wr, wi, info = sb03pd('X', 'N', 'N', a.copy(order='F'), c.copy(order='F'))

    assert info == 0 or info == n + 1
    assert scale > 0 and scale <= 1.0

    # Validate Lyapunov equation residual: A' * X * A - X = scale * C
    residual = a.T @ x @ a - x - scale * c
    assert np.allclose(residual, 0.0, atol=1e-12)

    # Validate solution symmetry: X should be symmetric
    assert np.allclose(x, x.T, rtol=1e-14)


def test_sb03pd_transpose_form():
    """
    Test discrete Lyapunov with TRANA='T' (A' * X * A - X = scale * C).

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import sb03pd

    np.random.seed(123)
    n = 4

    # Create a stable discrete-time matrix
    a = np.array([
        [0.4, 0.2, 0.0, 0.0],
        [0.0, 0.5, 0.1, 0.0],
        [0.0, 0.0, 0.3, 0.2],
        [0.0, 0.0, 0.0, 0.6]
    ], order='F', dtype=float)

    # Symmetric C
    temp = np.random.randn(n, n)
    c = (temp + temp.T) / 2
    c = np.asarray(c, order='F', dtype=float)

    # Solve with transpose form: op(A) = A^T
    # Equation becomes: A * X * A' - X = scale * C
    a_out, u, x, scale, wr, wi, info = sb03pd('X', 'N', 'T', a.copy(order='F'), c.copy(order='F'))

    assert info == 0 or info == n + 1
    assert scale > 0 and scale <= 1.0

    # For TRANA='T': op(A) = A', so equation is A * X * A' - X = scale * C
    residual = a @ x @ a.T - x - scale * c
    assert np.allclose(residual, 0.0, atol=1e-12)

    # Solution should be symmetric
    assert np.allclose(x, x.T, rtol=1e-14)


def test_sb03pd_with_separation():
    """
    Test separation estimation with JOB='S'.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sb03pd

    np.random.seed(456)
    n = 3

    # Stable matrix
    a = np.array([
        [0.3, 0.1, 0.0],
        [0.0, 0.4, 0.1],
        [0.0, 0.0, 0.2]
    ], order='F', dtype=float)

    # Dummy C (not used when JOB='S')
    c = np.zeros((n, n), order='F', dtype=float)

    # Compute separation only
    a_out, u, x_out, scale, sepd, wr, wi, info = sb03pd('S', 'N', 'N', a.copy(order='F'), c)

    assert info == 0 or info == n + 1
    assert sepd > 0.0  # Separation should be positive for stable system


def test_sb03pd_solution_and_separation():
    """
    Test both solution and separation with JOB='B'.

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import sb03pd

    np.random.seed(789)
    n = 3

    # Stable matrix
    a = np.array([
        [0.5, 0.0, 0.0],
        [0.1, 0.4, 0.0],
        [0.0, 0.2, 0.3]
    ], order='F', dtype=float)

    # Symmetric C
    c = np.array([
        [2.0, 0.5, 0.1],
        [0.5, 1.5, 0.2],
        [0.1, 0.2, 1.0]
    ], order='F', dtype=float)

    # Compute both solution and separation
    a_out, u, x, scale, sepd, ferr, wr, wi, info = sb03pd('B', 'N', 'N', a.copy(order='F'), c.copy(order='F'))

    assert info == 0 or info == n + 1
    assert scale > 0 and scale <= 1.0
    assert sepd > 0.0  # Separation should be positive
    assert ferr >= 0.0  # Forward error bound should be non-negative

    # Validate Lyapunov equation residual
    residual = a.T @ x @ a - x - scale * c
    assert np.allclose(residual, 0.0, atol=1e-12)

    # Solution should be symmetric
    assert np.allclose(x, x.T, rtol=1e-14)


def test_sb03pd_with_factored_input():
    """
    Test with pre-factored Schur form (FACT='F').

    When FACT='F', A should already be in Schur form and U is the orthogonal
    transformation matrix.

    Uses upper triangular T (already in Schur form) with U=I.
    """
    from ctrlsys import sb03pd

    n = 3

    # Upper triangular matrix is already in Schur form
    # Eigenvalues on diagonal must be inside unit circle for stability
    t = np.array([
        [0.3, 0.2, 0.1],
        [0.0, 0.4, 0.2],
        [0.0, 0.0, 0.5]
    ], order='F', dtype=float)

    # U = I since T is already Schur form of itself
    u = np.eye(n, order='F', dtype=float)

    # Symmetric C
    c = np.array([
        [1.0, 0.1, 0.0],
        [0.1, 1.5, 0.2],
        [0.0, 0.2, 2.0]
    ], order='F', dtype=float)

    # Solve with pre-factored input
    a_out, u_out, x, scale, wr, wi, info = sb03pd('X', 'F', 'N', t.copy(order='F'), c.copy(order='F'), u.copy(order='F'))

    assert info == 0 or info == n + 1
    assert scale > 0 and scale <= 1.0

    # Validate: T' * X * T - X = scale * C (T is both A and its Schur form)
    residual = t.T @ x @ t - x - scale * c
    assert np.allclose(residual, 0.0, atol=1e-11)


def test_sb03pd_zero_dimension():
    """
    Test quick return for N=0.
    """
    from ctrlsys import sb03pd

    n = 0
    a = np.zeros((0, 0), order='F', dtype=float)
    c = np.zeros((0, 0), order='F', dtype=float)

    a_out, u, x, scale, wr, wi, info = sb03pd('X', 'N', 'N', a, c)

    assert info == 0
    assert scale == 1.0


def test_sb03pd_eigenvalue_output():
    """
    Test that eigenvalues are computed when FACT='N'.

    Random seed: 202 (for reproducibility)
    """
    from ctrlsys import sb03pd

    np.random.seed(202)
    n = 3

    # Upper triangular stable matrix (eigenvalues on diagonal)
    a = np.array([
        [0.3, 0.1, 0.2],
        [0.0, 0.5, 0.1],
        [0.0, 0.0, 0.4]
    ], order='F', dtype=float)

    c = np.eye(n, order='F', dtype=float)

    a_out, u, x, scale, wr, wi, info = sb03pd('X', 'N', 'N', a.copy(order='F'), c.copy(order='F'))

    assert info == 0 or info == n + 1

    # Eigenvalues from wr, wi should match original eigenvalues
    eigs_computed = wr + 1j * wi
    eigs_expected = np.linalg.eigvals(a)

    # Sort by real part for comparison
    eigs_computed_sorted = np.sort(np.real(eigs_computed))
    eigs_expected_sorted = np.sort(np.real(eigs_expected))

    assert np.allclose(eigs_computed_sorted, eigs_expected_sorted, rtol=1e-12)


def test_sb03pd_complex_eigenvalue_matrix():
    """
    Test with matrix having complex conjugate eigenvalues (2x2 block).

    Random seed: 303 (for reproducibility)
    """
    from ctrlsys import sb03pd

    np.random.seed(303)
    n = 3

    # Matrix with a 2x2 block (complex eigenvalue pair) and one real eigenvalue
    # Eigenvalues: 0.3 +/- 0.2i (from 2x2 block), 0.4 (real)
    a = np.array([
        [0.3, 0.2, 0.1],
        [-0.2, 0.3, 0.0],
        [0.0, 0.0, 0.4]
    ], order='F', dtype=float)

    # Verify eigenvalues are inside unit circle
    eigs = np.linalg.eigvals(a)
    assert all(np.abs(eigs) < 1.0)

    c = np.eye(n, order='F', dtype=float)

    a_out, u, x, scale, wr, wi, info = sb03pd('X', 'N', 'N', a.copy(order='F'), c.copy(order='F'))

    assert info == 0 or info == n + 1
    assert scale > 0 and scale <= 1.0

    # Validate Lyapunov equation
    residual = a.T @ x @ a - x - scale * c
    assert np.allclose(residual, 0.0, atol=1e-11)

    # Solution should be symmetric
    assert np.allclose(x, x.T, rtol=1e-14)
