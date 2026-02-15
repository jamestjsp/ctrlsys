"""
Tests for SG03AD: Generalized Lyapunov/Stein equation solver for descriptor systems.

Solves:
- Continuous: op(A)' * X * op(E) + op(E)' * X * op(A) = SCALE * Y
- Discrete:   op(A)' * X * op(A) - op(E)' * X * op(E) = SCALE * Y

where op(M) = M or M^T.

Test data from SLICOT HTML documentation example.

Mathematical properties tested:
- Lyapunov residual equation
- Solution symmetry
- Separation and forward error estimation

Random seeds: 42, 123, 456, 789 (for reproducibility)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


"""Tests based on SLICOT HTML documentation example."""

def test_continuous_basic():
    """
    Validate basic functionality using SLICOT HTML doc example.

    Continuous-time generalized Lyapunov equation.
    N=3, JOB='B', DICO='C', FACT='N', TRANS='N', UPLO='U'
    """
    from ctrlsys import sg03ad

    n = 3

    a = np.array([
        [3.0, 1.0, 1.0],
        [1.0, 3.0, 0.0],
        [1.0, 0.0, 2.0]
    ], order='F', dtype=float)

    e = np.array([
        [1.0, 3.0, 0.0],
        [3.0, 2.0, 1.0],
        [1.0, 0.0, 1.0]
    ], order='F', dtype=float)

    x = np.array([
        [-64.0, -73.0, -28.0],
        [  0.0, -70.0, -25.0],
        [  0.0,   0.0, -18.0]
    ], order='F', dtype=float)

    result = sg03ad(
        'C', 'B', 'N', 'N', 'U', n,
        a.copy(order='F'),
        e.copy(order='F'),
        x.copy(order='F')
    )

    x_out, scale, sep, ferr, alphar, alphai, beta, a_out, e_out, q, z, info = result

    assert info == 0

    x_expected = np.array([
        [-2.0, -1.0,  0.0],
        [-1.0, -3.0, -1.0],
        [ 0.0, -1.0, -3.0]
    ], order='F', dtype=float)

    assert_allclose(x_out, x_expected, rtol=1e-3, atol=1e-3)

    assert scale > 0 and scale <= 1.0
    assert sep > 0

    assert_allclose(x_out, x_out.T, atol=1e-10)


"""Tests for continuous-time generalized Lyapunov equation."""

def test_continuous_identity_e():
    """
    Test continuous generalized Lyapunov with identity E.

    This reduces to standard continuous-time Lyapunov equation.
    A' * X + X * A = scale * Y
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sg03ad

    np.random.seed(42)
    n = 3

    a = np.array([
        [-1.0, 0.5, 0.2],
        [0.0, -2.0, 0.3],
        [0.0, 0.0, -1.5]
    ], order='F', dtype=float)

    e = np.eye(n, order='F', dtype=float)

    y = np.array([
        [1.0, 0.2, 0.1],
        [0.2, 2.0, 0.3],
        [0.1, 0.3, 1.5]
    ], order='F', dtype=float)

    a_orig = a.copy()

    result = sg03ad(
        'C', 'X', 'N', 'N', 'U', n,
        a.copy(order='F'),
        e.copy(order='F'),
        y.copy(order='F')
    )

    x_out, scale, sep, ferr, alphar, alphai, beta, a_out, e_out, q, z, info = result

    assert info == 0
    assert scale > 0

    residual = a_orig.T @ x_out + x_out @ a_orig - scale * y
    assert_allclose(residual, np.zeros_like(residual), atol=1e-10)

    assert_allclose(x_out, x_out.T, atol=1e-12)

def test_continuous_transpose():
    """
    Test continuous generalized Lyapunov with transpose option.

    A * X * E' + E * X * A' = scale * Y
    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import sg03ad

    np.random.seed(123)
    n = 3

    a = np.array([
        [-1.0, 0.5, 0.2],
        [0.0, -2.0, 0.3],
        [0.0, 0.0, -1.5]
    ], order='F', dtype=float)

    e = np.eye(n, order='F', dtype=float)

    y = np.array([
        [1.0, 0.2, 0.1],
        [0.2, 2.0, 0.3],
        [0.1, 0.3, 1.5]
    ], order='F', dtype=float)

    a_orig = a.copy()

    result = sg03ad(
        'C', 'X', 'N', 'T', 'U', n,
        a.copy(order='F'),
        e.copy(order='F'),
        y.copy(order='F')
    )

    x_out, scale, sep, ferr, alphar, alphai, beta, a_out, e_out, q, z, info = result

    assert info == 0

    residual = a_orig @ x_out + x_out @ a_orig.T - scale * y
    assert_allclose(residual, np.zeros_like(residual), atol=1e-10)


"""Tests for discrete-time generalized Lyapunov equation."""

def test_discrete_identity_e():
    """
    Test discrete generalized Lyapunov with identity E.

    A' * X * A - X = scale * Y
    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sg03ad

    np.random.seed(456)
    n = 3

    a = np.array([
        [0.5, 0.1, 0.0],
        [0.0, 0.4, 0.1],
        [0.0, 0.0, 0.3]
    ], order='F', dtype=float)

    e = np.eye(n, order='F', dtype=float)

    y = np.array([
        [1.0, 0.3, 0.2],
        [0.3, 2.0, 0.1],
        [0.2, 0.1, 1.5]
    ], order='F', dtype=float)

    a_orig = a.copy()

    result = sg03ad(
        'D', 'X', 'N', 'N', 'U', n,
        a.copy(order='F'),
        e.copy(order='F'),
        y.copy(order='F')
    )

    x_out, scale, sep, ferr, alphar, alphai, beta, a_out, e_out, q, z, info = result

    assert info == 0

    residual = a_orig.T @ x_out @ a_orig - x_out - scale * y
    assert_allclose(residual, np.zeros_like(residual), atol=1e-10)

    assert_allclose(x_out, x_out.T, atol=1e-12)

def test_discrete_transpose():
    """
    Test discrete generalized Lyapunov with transpose option.

    A * X * A' - E * X * E' = scale * Y
    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import sg03ad

    np.random.seed(789)
    n = 2

    a = np.array([
        [0.6, 0.2],
        [0.0, 0.5]
    ], order='F', dtype=float)

    e = np.eye(n, order='F', dtype=float)

    y = np.array([
        [1.0, 0.5],
        [0.5, 2.0]
    ], order='F', dtype=float)

    a_orig = a.copy()

    result = sg03ad(
        'D', 'X', 'N', 'T', 'U', n,
        a.copy(order='F'),
        e.copy(order='F'),
        y.copy(order='F')
    )

    x_out, scale, sep, ferr, alphar, alphai, beta, a_out, e_out, q, z, info = result

    assert info == 0

    residual = a_orig @ x_out @ a_orig.T - x_out - scale * y
    assert_allclose(residual, np.zeros_like(residual), atol=1e-10)


"""Mathematical property validation tests."""

def test_solution_symmetry():
    """
    Verify solution X is symmetric.

    Random seed: 111 (for reproducibility)
    """
    from ctrlsys import sg03ad

    np.random.seed(111)
    n = 4

    a = -np.eye(n) + 0.1 * np.random.randn(n, n)
    a = np.asfortranarray(a, dtype=float)

    e = np.eye(n, order='F', dtype=float)

    y = np.random.randn(n, n)
    y = y + y.T
    y = np.asfortranarray(y, dtype=float)

    result = sg03ad(
        'C', 'X', 'N', 'N', 'U', n,
        a.copy(order='F'),
        e.copy(order='F'),
        y.copy(order='F')
    )

    x_out, scale, sep, ferr, alphar, alphai, beta, a_out, e_out, q, z, info = result

    assert info == 0 or info == 4

    assert_allclose(x_out, x_out.T, atol=1e-12)

def test_schur_factorization_validity():
    """
    Verify Schur factorization: A = Q * A_s * Z' and E = Q * E_s * Z'.

    Random seed: 222 (for reproducibility)
    """
    from ctrlsys import sg03ad

    np.random.seed(222)
    n = 3

    a = np.array([
        [-1.0, 0.5, 0.2],
        [-0.3, -2.0, 0.3],
        [0.1, -0.2, -1.5]
    ], order='F', dtype=float)

    e = np.array([
        [1.0, 0.1, 0.0],
        [0.0, 1.0, 0.2],
        [0.0, 0.0, 1.0]
    ], order='F', dtype=float)

    y = np.array([
        [1.0, 0.2, 0.1],
        [0.2, 2.0, 0.3],
        [0.1, 0.3, 1.5]
    ], order='F', dtype=float)

    a_orig = a.copy()
    e_orig = e.copy()

    result = sg03ad(
        'C', 'X', 'N', 'N', 'U', n,
        a.copy(order='F'),
        e.copy(order='F'),
        y.copy(order='F')
    )

    x_out, scale, sep, ferr, alphar, alphai, beta, a_schur, e_schur, q, z, info = result

    assert info == 0

    assert_allclose(q.T @ q, np.eye(n), atol=1e-12)
    assert_allclose(z.T @ z, np.eye(n), atol=1e-12)

    a_reconstructed = q @ a_schur @ z.T
    e_reconstructed = q @ e_schur @ z.T
    assert_allclose(a_reconstructed, a_orig, atol=1e-10)
    assert_allclose(e_reconstructed, e_orig, atol=1e-10)

def test_eigenvalue_computation():
    """
    Verify eigenvalues of pencil (A, E) computed correctly.

    Random seed: 333 (for reproducibility)
    """
    from ctrlsys import sg03ad

    np.random.seed(333)
    n = 3

    a = np.array([
        [-1.0, 0.5, 0.2],
        [-0.3, -2.0, 0.3],
        [0.1, -0.2, -1.5]
    ], order='F', dtype=float)

    e = np.eye(n, order='F', dtype=float)

    y = np.array([
        [1.0, 0.2, 0.1],
        [0.2, 2.0, 0.3],
        [0.1, 0.3, 1.5]
    ], order='F', dtype=float)

    eigs_numpy = np.linalg.eigvals(a)

    result = sg03ad(
        'C', 'X', 'N', 'N', 'U', n,
        a.copy(order='F'),
        e.copy(order='F'),
        y.copy(order='F')
    )

    x_out, scale, sep, ferr, alphar, alphai, beta, a_out, e_out, q, z, info = result

    assert info == 0

    eigs_slicot = []
    for i in range(n):
        if abs(beta[i]) > 1e-14:
            eig = (alphar[i] + 1j * alphai[i]) / beta[i]
            eigs_slicot.append(eig)

    eigs_slicot_remaining = list(eigs_slicot)
    for e1 in eigs_numpy:
        dists = [abs(e1 - e2) for e2 in eigs_slicot_remaining]
        min_idx = np.argmin(dists)
        assert dists[min_idx] < 1e-10
        eigs_slicot_remaining.pop(min_idx)


"""Tests for separation estimation feature."""

def test_separation_only():
    """
    Test separation computation only (JOB='S').

    Random seed: 444 (for reproducibility)
    """
    from ctrlsys import sg03ad

    np.random.seed(444)
    n = 3

    a = np.array([
        [-1.0, 0.5, 0.2],
        [0.0, -2.0, 0.3],
        [0.0, 0.0, -1.5]
    ], order='F', dtype=float)

    e = np.eye(n, order='F', dtype=float)

    x = np.zeros((n, n), order='F', dtype=float)

    result = sg03ad(
        'C', 'S', 'N', 'N', 'U', n,
        a.copy(order='F'),
        e.copy(order='F'),
        x.copy(order='F')
    )

    x_out, scale, sep, ferr, alphar, alphai, beta, a_out, e_out, q, z, info = result

    assert info == 0
    assert sep > 0

def test_solution_and_separation():
    """
    Test both solution and separation (JOB='B').

    Random seed: 555 (for reproducibility)
    """
    from ctrlsys import sg03ad

    np.random.seed(555)
    n = 3

    a = np.array([
        [-1.0, 0.5, 0.2],
        [0.0, -2.0, 0.3],
        [0.0, 0.0, -1.5]
    ], order='F', dtype=float)

    e = np.eye(n, order='F', dtype=float)

    y = np.array([
        [1.0, 0.2, 0.1],
        [0.2, 2.0, 0.3],
        [0.1, 0.3, 1.5]
    ], order='F', dtype=float)

    a_orig = a.copy()

    result = sg03ad(
        'C', 'B', 'N', 'N', 'U', n,
        a.copy(order='F'),
        e.copy(order='F'),
        y.copy(order='F')
    )

    x_out, scale, sep, ferr, alphar, alphai, beta, a_out, e_out, q, z, info = result

    assert info == 0
    assert sep > 0
    assert ferr >= 0

    residual = a_orig.T @ x_out + x_out @ a_orig - scale * y
    assert_allclose(residual, np.zeros_like(residual), atol=1e-10)


"""Tests for factored input (FACT='F')."""

def test_factored_input():
    """
    Test with pre-computed Schur factorization (FACT='F').

    Random seed: 666 (for reproducibility)
    """
    from ctrlsys import sg03ad

    np.random.seed(666)
    n = 3

    a = np.array([
        [-1.0, 0.5, 0.2],
        [0.0, -2.0, 0.3],
        [0.0, 0.0, -1.5]
    ], order='F', dtype=float)

    e = np.eye(n, order='F', dtype=float)

    y1 = np.array([
        [1.0, 0.2, 0.1],
        [0.2, 2.0, 0.3],
        [0.1, 0.3, 1.5]
    ], order='F', dtype=float)

    result1 = sg03ad(
        'C', 'X', 'N', 'N', 'U', n,
        a.copy(order='F'),
        e.copy(order='F'),
        y1.copy(order='F')
    )

    x1, scale1, sep1, ferr1, alphar, alphai, beta, a_schur, e_schur, q, z, info = result1
    assert info == 0

    y2 = np.array([
        [2.0, 0.4, 0.2],
        [0.4, 3.0, 0.6],
        [0.2, 0.6, 2.5]
    ], order='F', dtype=float)

    result2 = sg03ad(
        'C', 'X', 'F', 'N', 'U', n,
        a_schur.copy(order='F'),
        e_schur.copy(order='F'),
        y2.copy(order='F'),
        q.copy(order='F'),
        z.copy(order='F')
    )

    x2, scale2, sep2, ferr2, alphar2, alphai2, beta2, a_out2, e_out2, q2, z2, info2 = result2

    assert info2 == 0

    a_orig = q @ a_schur @ z.T
    residual = a_orig.T @ x2 + x2 @ a_orig - scale2 * y2
    assert_allclose(residual, np.zeros_like(residual), atol=1e-9)


"""Edge case and error condition tests."""

def test_zero_dimension():
    """Test with n=0 (quick return)."""
    from ctrlsys import sg03ad

    n = 0

    a = np.array([], dtype=float).reshape(0, 0)
    e = np.array([], dtype=float).reshape(0, 0)
    x = np.array([], dtype=float).reshape(0, 0)

    result = sg03ad(
        'C', 'X', 'N', 'N', 'U', n,
        np.asfortranarray(a),
        np.asfortranarray(e),
        np.asfortranarray(x)
    )

    x_out, scale, sep, ferr, alphar, alphai, beta, a_out, e_out, q, z, info = result

    assert info == 0
    assert scale == 1.0

def test_small_system():
    """Test with n=1 (smallest non-trivial case)."""
    from ctrlsys import sg03ad

    n = 1

    a = np.array([[-2.0]], order='F', dtype=float)
    e = np.array([[1.0]], order='F', dtype=float)
    y = np.array([[4.0]], order='F', dtype=float)

    result = sg03ad(
        'C', 'X', 'N', 'N', 'U', n,
        a.copy(order='F'),
        e.copy(order='F'),
        y.copy(order='F')
    )

    x_out, scale, sep, ferr, alphar, alphai, beta, a_out, e_out, q, z, info = result

    assert info == 0
    assert x_out.shape == (1, 1)

    residual = a[0, 0] * x_out[0, 0] + x_out[0, 0] * a[0, 0] - scale * y[0, 0]
    assert abs(residual) < 1e-10

def test_complex_eigenvalues():
    """
    Test with complex eigenvalues in pencil (A, E).

    Random seed: 777 (for reproducibility)
    """
    from ctrlsys import sg03ad

    np.random.seed(777)
    n = 2

    a = np.array([
        [-1.0, 1.0],
        [-1.0, -1.0]
    ], order='F', dtype=float)

    e = np.eye(n, order='F', dtype=float)

    y = np.array([
        [1.0, 0.3],
        [0.3, 2.0]
    ], order='F', dtype=float)

    a_orig = a.copy()

    result = sg03ad(
        'C', 'X', 'N', 'N', 'U', n,
        a.copy(order='F'),
        e.copy(order='F'),
        y.copy(order='F')
    )

    x_out, scale, sep, ferr, alphar, alphai, beta, a_out, e_out, q, z, info = result

    assert info == 0

    residual = a_orig.T @ x_out + x_out @ a_orig - scale * y
    assert_allclose(residual, np.zeros_like(residual), atol=1e-11)

    assert_allclose(x_out, x_out.T, atol=1e-12)
