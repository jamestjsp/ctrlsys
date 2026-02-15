"""
Tests for SB03TD: Continuous-time Lyapunov equation solver with condition/error estimation.

Solves: op(A)'*X + X*op(A) = scale*C

Test data from SLICOT HTML documentation.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_sb03td_basic():
    """
    Test SB03TD with HTML doc example data.

    N=3, JOB='A' (compute all), FACT='N', TRANA='N', UPLO='U', LYAPUN='O'
    """
    from ctrlsys import sb03td

    n = 3
    # A matrix (read row-wise from HTML)
    a = np.array([
        [3.0, 1.0, 1.0],
        [1.0, 3.0, 0.0],
        [0.0, 0.0, 3.0]
    ], dtype=float, order='F')

    # C matrix (symmetric, read row-wise from HTML, upper part)
    c = np.array([
        [25.0, 24.0, 15.0],
        [24.0, 32.0,  8.0],
        [15.0,  8.0, 40.0]
    ], dtype=float, order='F')

    # Expected solution X (read row-wise from HTML)
    x_expected = np.array([
        [3.2604, 2.7187, 1.8616],
        [2.7187, 4.4271, 0.5699],
        [1.8616, 0.5699, 6.0461]
    ], dtype=float, order='F')

    x, t, u, wr, wi, scale, sep, rcond, ferr, info = sb03td(
        'A', 'N', 'N', 'U', 'O', a, c
    )

    assert info == 0
    assert_allclose(scale, 1.0, rtol=1e-4)
    assert_allclose(x, x_expected, rtol=1e-3, atol=1e-4)
    assert_allclose(sep, 4.9068, rtol=1e-3, atol=1e-4)
    assert_allclose(rcond, 0.3611, rtol=1e-3, atol=1e-4)
    assert ferr < 1e-10  # Error bound should be very small


def test_sb03td_solution_only():
    """
    Test SB03TD with JOB='X' (solution only).

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sb03td

    np.random.seed(42)
    n = 4

    # Create a stable matrix (all eigenvalues have negative real parts)
    a_rand = np.random.randn(n, n)
    a = a_rand - 3.0 * np.eye(n)  # Shift to make stable
    a = np.asfortranarray(a)

    # Create symmetric positive definite C
    c_rand = np.random.randn(n, n)
    c = c_rand @ c_rand.T
    c = np.asfortranarray(c)

    x, t, u, wr, wi, scale, sep, rcond, ferr, info = sb03td(
        'X', 'N', 'N', 'U', 'O', a, c
    )

    assert info == 0
    assert scale > 0.0
    assert scale <= 1.0

    # Verify solution satisfies Lyapunov equation: A'X + XA = scale*C
    residual = a.T @ x + x @ a - scale * c
    assert_allclose(residual, np.zeros((n, n)), atol=1e-10)


def test_sb03td_transpose():
    """
    Test SB03TD with TRANA='T' (transpose form).

    Solves: A*X + X*A' = scale*C
    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import sb03td

    np.random.seed(123)
    n = 3

    # Create a stable matrix
    a_rand = np.random.randn(n, n)
    a = a_rand - 4.0 * np.eye(n)
    a = np.asfortranarray(a)

    # Create symmetric positive definite C
    c_rand = np.random.randn(n, n)
    c = c_rand @ c_rand.T
    c = np.asfortranarray(c)

    x, t, u, wr, wi, scale, sep, rcond, ferr, info = sb03td(
        'X', 'N', 'T', 'U', 'O', a, c
    )

    assert info == 0

    # Verify solution satisfies transposed Lyapunov equation: A*X + X*A' = scale*C
    residual = a @ x + x @ a.T - scale * c
    assert_allclose(residual, np.zeros((n, n)), atol=1e-10)


def test_sb03td_lower_triangle():
    """
    Test SB03TD with UPLO='L' (lower triangular input).

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sb03td

    np.random.seed(456)
    n = 3

    # Create a stable matrix
    a_rand = np.random.randn(n, n)
    a = a_rand - 3.5 * np.eye(n)
    a = np.asfortranarray(a)

    # Create symmetric positive definite C
    c_rand = np.random.randn(n, n)
    c = c_rand @ c_rand.T
    c = np.asfortranarray(c)

    x, t, u, wr, wi, scale, sep, rcond, ferr, info = sb03td(
        'X', 'N', 'N', 'L', 'O', a, c
    )

    assert info == 0

    # Verify solution satisfies Lyapunov equation
    residual = a.T @ x + x @ a - scale * c
    assert_allclose(residual, np.zeros((n, n)), atol=1e-10)


def test_sb03td_reduced_lyapunov():
    """
    Test SB03TD with LYAPUN='R' (reduced Lyapunov equation with Schur form).

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import sb03td

    np.random.seed(789)
    n = 3

    # Create a quasi-upper-triangular matrix (Schur form)
    # Using a simple upper triangular matrix for simplicity
    t_in = np.array([
        [-2.0, 0.5, 0.3],
        [0.0, -3.0, 0.4],
        [0.0, 0.0, -4.0]
    ], dtype=float, order='F')

    # Create symmetric positive definite C
    c_rand = np.random.randn(n, n)
    c = c_rand @ c_rand.T
    c = np.asfortranarray(c)

    # For LYAPUN='R', we pass T directly as Schur form (FACT='F')
    # Need dummy A (won't be used) - but for FACT='N' we compute Schur
    x, t_out, u, wr, wi, scale, sep, rcond, ferr, info = sb03td(
        'X', 'F', 'N', 'U', 'R', t_in, c, t=t_in
    )

    assert info == 0

    # Verify solution satisfies Lyapunov equation with T: T'X + XT = scale*C
    residual = t_in.T @ x + x @ t_in - scale * c
    assert_allclose(residual, np.zeros((n, n)), atol=1e-10)


def test_sb03td_condition_number():
    """
    Test SB03TD with JOB='A' (compute all including condition number).

    Random seed: 111 (for reproducibility)
    """
    from ctrlsys import sb03td

    np.random.seed(111)
    n = 3

    # Create a stable matrix
    a_rand = np.random.randn(n, n)
    a = a_rand - 3.0 * np.eye(n)
    a = np.asfortranarray(a)

    # Create symmetric positive definite C
    c_rand = np.random.randn(n, n)
    c = c_rand @ c_rand.T
    c = np.asfortranarray(c)

    # Compute all (solution, separation, condition, error)
    x, t, u, wr, wi, scale, sep, rcond, ferr, info = sb03td(
        'A', 'N', 'N', 'U', 'O', a, c
    )

    assert info == 0
    assert scale > 0.0
    assert sep > 0.0
    assert rcond > 0.0
    assert rcond <= 1.0
    assert ferr >= 0.0


def test_sb03td_separation_only():
    """
    Test SB03TD with JOB='S' (separation only).

    Random seed: 222 (for reproducibility)
    """
    from ctrlsys import sb03td

    np.random.seed(222)
    n = 3

    # Create a stable matrix
    a_rand = np.random.randn(n, n)
    a = a_rand - 3.0 * np.eye(n)
    a = np.asfortranarray(a)

    # For JOB='S', C is not referenced
    c = np.zeros((n, n), dtype=float, order='F')

    x, t, u, wr, wi, scale, sep, rcond, ferr, info = sb03td(
        'S', 'N', 'N', 'U', 'O', a, c
    )

    assert info == 0
    assert sep > 0.0  # Separation should be positive for stable matrix


def test_sb03td_error_bound():
    """
    Test SB03TD with JOB='A' including error bound.

    Random seed: 333 (for reproducibility)
    """
    from ctrlsys import sb03td

    np.random.seed(333)
    n = 3

    # Create a stable matrix
    a_rand = np.random.randn(n, n)
    a = a_rand - 3.0 * np.eye(n)
    a = np.asfortranarray(a)

    # Create symmetric positive definite C
    c_rand = np.random.randn(n, n)
    c = c_rand @ c_rand.T
    c = np.asfortranarray(c)

    # Compute all including error bound
    x, t, u, wr, wi, scale, sep, rcond, ferr, info = sb03td(
        'A', 'N', 'N', 'U', 'O', a, c
    )

    assert info == 0
    assert ferr >= 0.0  # Error bound should be non-negative

    # Verify Lyapunov equation
    residual = a.T @ x + x @ a - scale * c
    assert_allclose(residual, np.zeros((n, n)), atol=1e-10)


def test_sb03td_symmetry_preservation():
    """
    Test that SB03TD produces a symmetric solution matrix X.

    Random seed: 444 (for reproducibility)
    """
    from ctrlsys import sb03td

    np.random.seed(444)
    n = 5

    # Create a stable matrix
    a_rand = np.random.randn(n, n)
    a = a_rand - 4.0 * np.eye(n)
    a = np.asfortranarray(a)

    # Create symmetric positive definite C
    c_rand = np.random.randn(n, n)
    c = c_rand @ c_rand.T
    c = np.asfortranarray(c)

    x, t, u, wr, wi, scale, sep, rcond, ferr, info = sb03td(
        'X', 'N', 'N', 'U', 'O', a, c
    )

    assert info == 0
    # X should be symmetric
    assert_allclose(x, x.T, rtol=1e-14)


def test_sb03td_zero_dimension():
    """
    Test SB03TD with N=0 (quick return).
    """
    from ctrlsys import sb03td

    n = 0
    a = np.array([], dtype=float, order='F').reshape(0, 0)
    c = np.array([], dtype=float, order='F').reshape(0, 0)

    x, t, u, wr, wi, scale, sep, rcond, ferr, info = sb03td(
        'A', 'N', 'N', 'U', 'O', a, c
    )

    assert info == 0
    assert scale == 1.0
    assert rcond == 1.0
    assert ferr == 0.0


def test_sb03td_invalid_job():
    """
    Test SB03TD with invalid JOB parameter.
    """
    from ctrlsys import sb03td

    n = 3
    a = np.eye(n, dtype=float, order='F')
    c = np.eye(n, dtype=float, order='F')

    with pytest.raises(ValueError):
        sb03td('Z', 'N', 'N', 'U', 'O', a, c)
