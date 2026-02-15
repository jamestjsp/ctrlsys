"""
Tests for SB03OU: Solve Lyapunov equation for Cholesky factor.

Solves for U where X = op(U)'*op(U) satisfies:
  Continuous: op(A)'*X + X*op(A) = -scale^2 * op(B)'*op(B)
  Discrete:   op(A)'*X*op(A) - X = -scale^2 * op(B)'*op(B)

Tests:
1. Continuous-time with stable diagonal A
2. Discrete-time with convergent A
3. Lyapunov residual verification
4. Non-transpose case
5. Property: X = U'*U is positive semi-definite

Random seeds: 42, 123 (for reproducibility)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_sb03ou_continuous_basic():
    """
    Validate continuous-time Lyapunov with diagonal stable A.

    A is stable (negative eigenvalues), B is identity.
    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import sb03ou

    n = 3
    m = 3

    # Stable diagonal A
    a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')

    # B = identity
    b = np.eye(m, n, dtype=float, order='F')

    u, scale, info = sb03ou(False, False, a, b.copy(order='F'))

    assert info == 0 or info == 1  # 0=success, 1=nearly singular (warning)
    assert scale > 0
    assert u.shape == (n, n)

    # U should be upper triangular
    assert np.allclose(np.tril(u, -1), 0, atol=1e-14)

    # Verify Lyapunov equation residual: A'*X + X*A + scale^2*B'*B = 0
    x = u.T @ u
    residual = a.T @ x + x @ a + (scale**2) * (b.T @ b)
    assert_allclose(residual, np.zeros((n, n)), atol=1e-10)


def test_sb03ou_discrete_basic():
    """
    Validate discrete-time Lyapunov with convergent A.

    A has eigenvalues inside unit circle.
    """
    from ctrlsys import sb03ou

    n = 2
    m = 2

    # Convergent A (eigenvalues inside unit circle)
    a = np.array([
        [0.5, 0.1],
        [0.0, 0.3]
    ], order='F', dtype=float)

    # B = identity
    b = np.eye(m, n, dtype=float, order='F')

    u, scale, info = sb03ou(True, False, a, b.copy(order='F'))

    assert info == 0 or info == 1
    assert scale > 0
    assert u.shape == (n, n)

    # U should be upper triangular
    assert np.allclose(np.tril(u, -1), 0, atol=1e-14)

    # Verify discrete Lyapunov: A'*X*A - X + scale^2*B'*B = 0
    x = u.T @ u
    residual = a.T @ x @ a - x + (scale**2) * (b.T @ b)
    assert_allclose(residual, np.zeros((n, n)), atol=1e-10)


def test_sb03ou_transpose():
    """
    Validate with transpose option (LTRANS=True).

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import sb03ou

    np.random.seed(123)
    n = 3
    m = 3

    # Stable upper triangular A
    a = np.triu(np.random.randn(n, n))
    for i in range(n):
        a[i, i] = -abs(a[i, i]) - 0.5
    a = np.asfortranarray(a)

    # B (n x m for LTRANS=True)
    b = np.random.randn(n, m).astype(float, order='F')

    u, scale, info = sb03ou(False, True, a, b.copy(order='F'))

    assert info == 0 or info == 1
    assert scale > 0

    # For LTRANS=True: X = U*U' and op(K) = K'
    x = u @ u.T
    # A*X + X*A' + scale^2*B*B' = 0
    residual = a @ x + x @ a.T + (scale**2) * (b @ b.T)
    assert_allclose(residual, np.zeros((n, n)), atol=1e-10)


def test_sb03ou_m_greater_n():
    """
    Validate with M > N (more rows in B than order of A).
    """
    from ctrlsys import sb03ou

    n = 2
    m = 4

    a = np.diag([-1.0, -2.0]).astype(float, order='F')
    b = np.array([
        [1.0, 0.5],
        [0.0, 1.0],
        [0.2, 0.3],
        [0.1, 0.4]
    ], order='F', dtype=float)

    u, scale, info = sb03ou(False, False, a, b.copy(order='F'))

    assert info == 0 or info == 1
    assert u.shape == (n, n)

    # Verify residual
    x = u.T @ u
    residual = a.T @ x + x @ a + (scale**2) * (b.T @ b)
    assert_allclose(residual, np.zeros((n, n)), atol=1e-10)


def test_sb03ou_unstable_error():
    """
    Validate error for unstable A (INFO=2).
    """
    from ctrlsys import sb03ou

    n = 2
    m = 2

    # Unstable A (positive eigenvalue for continuous)
    a = np.array([
        [1.0, 0.0],
        [0.0, -1.0]
    ], order='F', dtype=float)

    b = np.eye(m, n, dtype=float, order='F')

    u, scale, info = sb03ou(False, False, a, b.copy(order='F'))

    assert info == 2  # Unstable


def test_sb03ou_positive_semidefinite():
    """
    Validate X = U'*U is positive semi-definite.

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import sb03ou

    np.random.seed(456)
    n = 4
    m = 4

    # Create stable Schur form A
    a = np.triu(np.random.randn(n, n))
    for i in range(n):
        a[i, i] = -abs(a[i, i]) - 0.5
    a = np.asfortranarray(a)

    b = np.random.randn(m, n).astype(float, order='F')

    u, scale, info = sb03ou(False, False, a, b.copy(order='F'))

    if info == 0 or info == 1:
        x = u.T @ u
        eig = np.linalg.eigvalsh(x)
        assert all(e >= -1e-10 for e in eig), "X not positive semi-definite"
