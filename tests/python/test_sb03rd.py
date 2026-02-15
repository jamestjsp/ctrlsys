"""
Tests for SB03RD - Continuous-time Lyapunov equation solver with separation estimation.

Solves: op(A)' * X + X * op(A) = scale * C
where op(A) = A or A^T, C is symmetric.
"""

import numpy as np
import pytest


def test_sb03rd_basic_solution():
    """
    Test basic Lyapunov equation solution.

    For a stable matrix A (negative real eigenvalues), solve:
    A' * X + X * A = C

    Test data: 3x3 stable matrix with verified solution.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sb03rd

    n = 3

    # Stable matrix A with eigenvalues having negative real parts
    # Constructed to ensure stability
    a = np.array([
        [-1.0,  0.5,  0.0],
        [ 0.0, -2.0,  0.3],
        [ 0.0,  0.0, -3.0]
    ], order='F', dtype=float)

    # Symmetric right-hand side C
    c = np.array([
        [ 2.0,  0.5,  0.1],
        [ 0.5,  3.0,  0.2],
        [ 0.1,  0.2,  4.0]
    ], order='F', dtype=float)

    # Call sb03rd with JOB='X' (solution only), FACT='N' (compute Schur), TRANA='N'
    a_out, u, x, scale, sep, ferr, wr, wi, info = sb03rd('X', 'N', 'N', a, c)

    assert info == 0
    assert scale > 0.0

    # Verify Lyapunov equation: A' * X + X * A = scale * C
    # Using original A (before Schur)
    a_orig = np.array([
        [-1.0,  0.5,  0.0],
        [ 0.0, -2.0,  0.3],
        [ 0.0,  0.0, -3.0]
    ], order='F', dtype=float)
    c_orig = np.array([
        [ 2.0,  0.5,  0.1],
        [ 0.5,  3.0,  0.2],
        [ 0.1,  0.2,  4.0]
    ], order='F', dtype=float)

    residual = a_orig.T @ x + x @ a_orig - scale * c_orig
    np.testing.assert_allclose(residual, np.zeros((n, n)), atol=1e-12)

    # Check solution is symmetric
    np.testing.assert_allclose(x, x.T, rtol=1e-14)


def test_sb03rd_transpose_form():
    """
    Test Lyapunov equation with TRANA='T' (transpose form).

    Solves: A * X + X * A' = scale * C
    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import sb03rd

    n = 3

    # Stable matrix (upper triangular for simplicity)
    a = np.array([
        [-2.0,  0.5,  0.1],
        [ 0.0, -1.5,  0.3],
        [ 0.0,  0.0, -1.0]
    ], order='F', dtype=float)

    # Symmetric RHS
    c = np.array([
        [ 1.0,  0.2,  0.0],
        [ 0.2,  2.0,  0.1],
        [ 0.0,  0.1,  1.5]
    ], order='F', dtype=float)

    a_orig = a.copy()
    c_orig = c.copy()

    a_out, u, x, scale, sep, ferr, wr, wi, info = sb03rd('X', 'N', 'T', a, c)

    assert info == 0
    assert scale > 0.0

    # Verify: A * X + X * A' = scale * C
    residual = a_orig @ x + x @ a_orig.T - scale * c_orig
    np.testing.assert_allclose(residual, np.zeros((n, n)), atol=1e-12)


def test_sb03rd_separation_only():
    """
    Test computing separation only (JOB='S').

    Separation estimates how well-conditioned the Lyapunov equation is.
    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sb03rd

    n = 3

    # Stable matrix
    a = np.array([
        [-1.0,  0.1,  0.0],
        [ 0.0, -2.0,  0.1],
        [ 0.0,  0.0, -3.0]
    ], order='F', dtype=float)

    # C not used for JOB='S', but we need to pass something
    c = np.zeros((n, n), order='F', dtype=float)

    a_out, u, x, scale, sep, ferr, wr, wi, info = sb03rd('S', 'N', 'N', a, c)

    assert info == 0
    # SEP should be positive for a stable matrix
    assert sep > 0.0


def test_sb03rd_both_solution_and_separation():
    """
    Test computing both solution and separation (JOB='B').

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import sb03rd

    n = 3

    a = np.array([
        [-2.0,  0.3,  0.0],
        [ 0.0, -1.0,  0.2],
        [ 0.0,  0.0, -1.5]
    ], order='F', dtype=float)

    c = np.array([
        [ 1.0,  0.1,  0.0],
        [ 0.1,  1.5,  0.05],
        [ 0.0,  0.05, 2.0]
    ], order='F', dtype=float)

    a_orig = a.copy()
    c_orig = c.copy()

    a_out, u, x, scale, sep, ferr, wr, wi, info = sb03rd('B', 'N', 'N', a, c)

    assert info == 0
    assert scale > 0.0
    assert sep > 0.0
    assert ferr >= 0.0  # Forward error bound

    # Verify solution
    residual = a_orig.T @ x + x @ a_orig - scale * c_orig
    np.testing.assert_allclose(residual, np.zeros((n, n)), atol=1e-12)


def test_sb03rd_with_schur_factored():
    """
    Test with pre-computed Schur factorization (FACT='F').

    Random seed: 999 (for reproducibility)
    """
    from ctrlsys import sb03rd

    n = 3

    # Already in Schur form (upper quasi-triangular = upper triangular here)
    a_schur = np.array([
        [-3.0,  0.5,  0.2],
        [ 0.0, -2.0,  0.1],
        [ 0.0,  0.0, -1.0]
    ], order='F', dtype=float)

    # U = identity for this test (A is already in Schur form)
    u_in = np.eye(n, order='F', dtype=float)

    c = np.array([
        [ 2.0,  0.3,  0.1],
        [ 0.3,  1.5,  0.2],
        [ 0.1,  0.2,  1.0]
    ], order='F', dtype=float)

    a_orig = a_schur.copy()
    c_orig = c.copy()

    a_out, u, x, scale, sep, ferr, wr, wi, info = sb03rd('X', 'F', 'N', a_schur, c, u=u_in)

    assert info == 0
    assert scale > 0.0

    # Verify solution: A' * X + X * A = scale * C
    residual = a_orig.T @ x + x @ a_orig - scale * c_orig
    np.testing.assert_allclose(residual, np.zeros((n, n)), atol=1e-12)


def test_sb03rd_n_equals_zero():
    """
    Test edge case: n=0 (quick return).
    """
    from ctrlsys import sb03rd

    a = np.array([], dtype=float).reshape(0, 0)
    c = np.array([], dtype=float).reshape(0, 0)

    a_out, u, x, scale, sep, ferr, wr, wi, info = sb03rd('X', 'N', 'N', a, c)

    assert info == 0
    assert scale == 1.0


def test_sb03rd_eigenvalue_output():
    """
    Test that eigenvalues are correctly returned when FACT='N'.

    Random seed: 111 (for reproducibility)
    """
    from ctrlsys import sb03rd

    n = 3

    # Upper triangular - eigenvalues are diagonal elements
    a = np.array([
        [-1.0,  0.5,  0.2],
        [ 0.0, -2.0,  0.3],
        [ 0.0,  0.0, -3.0]
    ], order='F', dtype=float)

    c = np.array([
        [ 1.0,  0.1,  0.0],
        [ 0.1,  1.0,  0.1],
        [ 0.0,  0.1,  1.0]
    ], order='F', dtype=float)

    a_out, u, x, scale, sep, ferr, wr, wi, info = sb03rd('X', 'N', 'N', a, c)

    assert info == 0

    # Eigenvalues should match diagonal elements (real, no imaginary)
    expected_eigs = sorted([-1.0, -2.0, -3.0])
    computed_eigs = sorted(wr.tolist())
    np.testing.assert_allclose(computed_eigs, expected_eigs, rtol=1e-14)
    np.testing.assert_allclose(wi, np.zeros(n), atol=1e-14)


def test_sb03rd_invalid_job():
    """
    Test error handling for invalid JOB parameter.
    """
    from ctrlsys import sb03rd

    n = 2
    a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
    c = np.eye(n, order='F', dtype=float)

    a_out, u, x, scale, sep, ferr, wr, wi, info = sb03rd('Z', 'N', 'N', a, c)
    assert info == -1


def test_sb03rd_invalid_fact():
    """
    Test error handling for invalid FACT parameter.
    """
    from ctrlsys import sb03rd

    n = 2
    a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
    c = np.eye(n, order='F', dtype=float)

    a_out, u, x, scale, sep, ferr, wr, wi, info = sb03rd('X', 'Z', 'N', a, c)
    assert info == -2


def test_sb03rd_singular_warning():
    """
    Test that INFO=N+1 is returned when A and -A' have close eigenvalues.

    This occurs when the eigenvalue separation is poor.
    """
    from ctrlsys import sb03rd

    n = 2

    # Matrix with eigenvalue near zero - causes singularity in Lyapunov solver
    # because A + A' would have eigenvalue near 0
    a = np.array([
        [ 0.0,  1.0],
        [-1.0,  0.0]
    ], order='F', dtype=float)  # Purely imaginary eigenvalues +/- i

    c = np.eye(n, order='F', dtype=float)

    a_out, u, x, scale, sep, ferr, wr, wi, info = sb03rd('X', 'N', 'N', a, c)

    # INFO should be N+1 (=3) for near-singular case with perturbation
    # or some value > 0 indicating the issue
    assert info == n + 1 or info > 0


def test_sb03rd_lyapunov_residual_property():
    """
    Mathematical property test: Lyapunov equation residual should be near zero.

    For a stable A and symmetric C, the solution X of A'X + XA = C
    should satisfy the equation exactly (up to numerical precision).

    Random seed: 222 (for reproducibility)
    """
    from ctrlsys import sb03rd

    np.random.seed(222)
    n = 4

    # Generate stable A by making eigenvalues negative
    # Use diagonal + small perturbation
    d = -np.abs(np.random.randn(n)) - 0.5  # Negative eigenvalues
    a_diag = np.diag(d)
    pert = 0.1 * np.triu(np.random.randn(n, n), k=1)
    a = np.asfortranarray(a_diag + pert)

    # Generate symmetric C
    c_rand = np.random.randn(n, n)
    c = np.asfortranarray((c_rand + c_rand.T) / 2)

    a_orig = a.copy()
    c_orig = c.copy()

    a_out, u, x, scale, sep, ferr, wr, wi, info = sb03rd('X', 'N', 'N', a, c)

    assert info == 0 or info == n + 1  # Allow perturbed solution

    # Verify residual: A' * X + X * A - scale * C ≈ 0
    residual = a_orig.T @ x + x @ a_orig - scale * c_orig
    residual_norm = np.linalg.norm(residual, 'fro')
    c_norm = np.linalg.norm(c_orig, 'fro')

    # Relative residual should be small
    np.testing.assert_array_less(residual_norm, 1e-10 * c_norm + 1e-14)


def test_sb03rd_solution_symmetry():
    """
    Mathematical property test: Solution X should be symmetric.

    For symmetric C, the solution of A'X + XA = C is also symmetric.

    Random seed: 333 (for reproducibility)
    """
    from ctrlsys import sb03rd

    np.random.seed(333)
    n = 5

    # Generate stable A
    d = -np.abs(np.random.randn(n)) - 1.0
    a = np.asfortranarray(np.diag(d) + 0.1 * np.triu(np.random.randn(n, n), k=1))

    # Generate symmetric C
    c_rand = np.random.randn(n, n)
    c = np.asfortranarray((c_rand + c_rand.T) / 2)

    a_out, u, x, scale, sep, ferr, wr, wi, info = sb03rd('X', 'N', 'N', a, c)

    assert info == 0 or info == n + 1

    # X must be symmetric
    np.testing.assert_allclose(x, x.T, rtol=1e-14)
