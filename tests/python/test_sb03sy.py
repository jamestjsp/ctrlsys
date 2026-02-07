"""
Tests for SB03SY: Estimate separation and Theta norm for discrete-time Lyapunov.

SB03SY estimates:
- sepd(op(A),op(A)') = min norm(op(A)'*X*op(A) - X)/norm(X)
- 1-norm of Theta operator

for discrete-time Lyapunov equation op(A)'*X*op(A) - X = C.
"""

import numpy as np
import pytest
from slicot import sb03sy


def test_sb03sy_separation_only():
    """
    Test separation estimation with JOB='S'.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 4

    t = np.zeros((n, n), order='F', dtype=float)
    for i in range(n):
        t[i, i] = 0.5 + 0.1 * i
        for j in range(i + 1, n):
            t[i, j] = np.random.randn() * 0.1

    u = np.eye(n, order='F', dtype=float)

    xa = np.eye(n, order='F', dtype=float)

    sepd, thnorm, info = sb03sy('S', 'N', 'R', n, t, u, xa)

    assert info >= 0
    assert sepd > 0.0

    kron_op = np.kron(t.T, t.T) - np.eye(n * n)
    true_sepd = np.linalg.svd(kron_op, compute_uv=False)[-1]
    assert sepd <= true_sepd * n + 1e-10


def test_sb03sy_theta_only():
    """
    Test Theta norm estimation with JOB='T'.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 3

    t = np.zeros((n, n), order='F', dtype=float)
    for i in range(n):
        t[i, i] = 0.4 + 0.15 * i
        for j in range(i + 1, n):
            t[i, j] = np.random.randn() * 0.1

    u = np.eye(n, order='F', dtype=float)

    xa = np.random.randn(n, n).astype(float, order='F')

    sepd, thnorm, info = sb03sy('T', 'N', 'R', n, t, u, xa)

    assert info >= 0
    assert thnorm >= 0.0
    assert thnorm < 1e10


def test_sb03sy_both():
    """
    Test computing both separation and Theta norm with JOB='B'.

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 3

    t = np.zeros((n, n), order='F', dtype=float)
    for i in range(n):
        t[i, i] = 0.3 + 0.2 * i
        for j in range(i + 1, n):
            t[i, j] = np.random.randn() * 0.1

    u = np.eye(n, order='F', dtype=float)
    xa = np.random.randn(n, n).astype(float, order='F')

    sepd, thnorm, info = sb03sy('B', 'N', 'R', n, t, u, xa)

    assert info >= 0
    assert sepd > 0.0
    assert thnorm >= 0.0

    sepd2, _, info2 = sb03sy('S', 'N', 'R', n, t, u, xa)
    assert info2 >= 0
    np.testing.assert_allclose(sepd, sepd2, rtol=1e-10)


def test_sb03sy_trans():
    """
    Test with TRANA='T'.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 3

    t = np.zeros((n, n), order='F', dtype=float)
    for i in range(n):
        t[i, i] = 0.4 + 0.15 * i
        for j in range(i + 1, n):
            t[i, j] = np.random.randn() * 0.1

    u = np.eye(n, order='F', dtype=float)
    xa = np.eye(n, order='F', dtype=float)

    sepd, thnorm, info = sb03sy('S', 'T', 'R', n, t, u, xa)

    assert info >= 0
    assert sepd > 0.0

    kron_op = np.kron(t, t) - np.eye(n * n)
    true_sepd = np.linalg.svd(kron_op, compute_uv=False)[-1]
    assert sepd <= true_sepd * n + 1e-10


def test_sb03sy_original_mode():
    """
    Test with LYAPUN='O' (original mode with orthogonal U).

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    n = 3

    t = np.zeros((n, n), order='F', dtype=float)
    for i in range(n):
        t[i, i] = 0.3 + 0.2 * i
        for j in range(i + 1, n):
            t[i, j] = np.random.randn() * 0.1

    q, _ = np.linalg.qr(np.random.randn(n, n))
    u = q.astype(float, order='F')

    xa = np.random.randn(n, n).astype(float, order='F')

    sepd, thnorm, info = sb03sy('B', 'N', 'O', n, t, u, xa)

    assert info >= 0
    assert sepd > 0.0
    assert thnorm >= 0.0
    assert thnorm < 1e10


def test_sb03sy_known_problem():
    n = 4
    diag_vals = np.array([0.5, 0.6, 0.7, 0.3])
    t = np.diag(diag_vals).astype(float, order='F')
    u = np.eye(n, order='F', dtype=float)

    x = np.eye(n, dtype=float)
    # XA = U'*X*U*op(T) = X*T for LYAPUN='R', TRANA='N'
    xa = np.asfortranarray(x @ t)

    sepd, thnorm, info = sb03sy('B', 'N', 'R', n, t, u, xa)

    assert info >= 0
    assert sepd > 0.0
    assert thnorm >= 0.0

    kron_op = np.kron(t.T, t.T) - np.eye(n * n)
    true_sepd = np.linalg.svd(kron_op, compute_uv=False)[-1]
    assert sepd <= true_sepd * n + 1e-10
    assert sepd >= true_sepd / n - 1e-10


def test_sb03sy_n_zero():
    """
    Test quick return for n=0
    """
    n = 0
    t = np.array([], order='F', dtype=float).reshape(0, 0)
    u = np.array([], order='F', dtype=float).reshape(0, 0)
    xa = np.array([], order='F', dtype=float).reshape(0, 0)

    sepd, thnorm, info = sb03sy('B', 'N', 'R', n, t, u, xa)

    assert info == 0


def test_sb03sy_invalid_job():
    """
    Test error handling for invalid JOB parameter
    """
    n = 2
    t = np.eye(n, order='F', dtype=float)
    u = np.eye(n, order='F', dtype=float)
    xa = np.eye(n, order='F', dtype=float)

    sepd, thnorm, info = sb03sy('X', 'N', 'R', n, t, u, xa)

    assert info == -1


def test_sb03sy_invalid_trana():
    """
    Test error handling for invalid TRANA parameter
    """
    n = 2
    t = np.eye(n, order='F', dtype=float)
    u = np.eye(n, order='F', dtype=float)
    xa = np.eye(n, order='F', dtype=float)

    sepd, thnorm, info = sb03sy('S', 'X', 'R', n, t, u, xa)

    assert info == -2


def test_sb03sy_invalid_lyapun():
    """
    Test error handling for invalid LYAPUN parameter
    """
    n = 2
    t = np.eye(n, order='F', dtype=float)
    u = np.eye(n, order='F', dtype=float)
    xa = np.eye(n, order='F', dtype=float)

    sepd, thnorm, info = sb03sy('S', 'N', 'X', n, t, u, xa)

    assert info == -3
