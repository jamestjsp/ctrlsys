"""
Tests for SB03SX: Estimate forward error for discrete-time Lyapunov equation.

SB03SX estimates a forward error bound for the solution X of:
    op(A)'*X*op(A) - X = C
where op(A) = A or A' and C is symmetric.
"""

import numpy as np
import pytest
from ctrlsys import sb03sx


def test_sb03sx_basic_notrans():
    """
    Test basic forward error estimation with TRANA='N', LYAPUN='R'.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 4

    t = np.zeros((n, n), order='F', dtype=float)
    for i in range(n):
        t[i, i] = 0.5 + 0.1 * i
        for j in range(i + 1, n):
            t[i, j] = np.random.randn() * 0.1

    xanorm = 1.5

    r = np.random.randn(n, n).astype(float, order='F')
    r = (r + r.T) / 2
    r = np.triu(r)

    u = np.eye(n, order='F', dtype=float)

    ferr, r_out, info = sb03sx('N', 'U', 'R', n, xanorm, t, u, r)

    assert info >= 0
    assert ferr >= 0.0
    assert ferr < 1e6


def test_sb03sx_trans():
    """
    Test with TRANA='T', LYAPUN='R'.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 3

    t = np.zeros((n, n), order='F', dtype=float)
    for i in range(n):
        t[i, i] = 0.4 + 0.15 * i
        for j in range(i + 1, n):
            t[i, j] = np.random.randn() * 0.1

    xanorm = 2.0

    r = np.random.randn(n, n).astype(float, order='F')
    r = (r + r.T) / 2
    r = np.tril(r)

    u = np.eye(n, order='F', dtype=float)

    ferr, r_out, info = sb03sx('T', 'L', 'R', n, xanorm, t, u, r)

    assert info >= 0
    assert ferr >= 0.0
    assert ferr < 1e6


def test_sb03sx_original_mode():
    """
    Test with LYAPUN='O' (original mode with orthogonal U).

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 3

    t = np.zeros((n, n), order='F', dtype=float)
    for i in range(n):
        t[i, i] = 0.3 + 0.2 * i
        for j in range(i + 1, n):
            t[i, j] = np.random.randn() * 0.1

    q, _ = np.linalg.qr(np.random.randn(n, n))
    u = q.astype(float, order='F')

    xanorm = 1.0

    r = np.random.randn(n, n).astype(float, order='F')
    r = (r + r.T) / 2
    r = np.triu(r)

    ferr, r_out, info = sb03sx('N', 'U', 'O', n, xanorm, t, u, r)

    assert info >= 0
    assert ferr >= 0.0
    assert ferr < 1e6


def test_sb03sx_known_solution():
    n = 4
    diag_vals = np.array([0.5, 0.6, 0.7, 0.3])
    t = np.diag(diag_vals).astype(float, order='F')
    u = np.eye(n, order='F', dtype=float)

    x = np.eye(n, dtype=float)
    c = t.T @ x @ t - x
    xanorm = np.max(np.abs(x))

    residual = t.T @ x @ t - x - c
    r = np.asfortranarray(np.abs(residual) + 1e-15 * xanorm * np.ones((n, n)))
    r = np.triu(r)

    ferr, r_out, info = sb03sx('N', 'U', 'R', n, xanorm, t, u, r)

    assert info >= 0
    assert ferr >= 0.0
    assert ferr < 1.0


def test_sb03sx_n_zero():
    """
    Test quick return for n=0
    """
    n = 0
    xanorm = 1.0

    t = np.array([], order='F', dtype=float).reshape(0, 0)
    u = np.array([], order='F', dtype=float).reshape(0, 0)
    r = np.array([], order='F', dtype=float).reshape(0, 0)

    ferr, r_out, info = sb03sx('N', 'U', 'R', n, xanorm, t, u, r)

    assert info == 0
    assert ferr == 0.0


def test_sb03sx_xanorm_zero():
    """
    Test quick return for xanorm=0
    """
    np.random.seed(789)
    n = 3
    xanorm = 0.0

    t = np.eye(n, order='F', dtype=float)
    u = np.eye(n, order='F', dtype=float)
    r = np.eye(n, order='F', dtype=float)

    ferr, r_out, info = sb03sx('N', 'U', 'R', n, xanorm, t, u, r)

    assert info == 0
    assert ferr == 0.0


def test_sb03sx_invalid_trana():
    """
    Test error handling for invalid TRANA parameter
    """
    n = 2
    t = np.eye(n, order='F', dtype=float)
    u = np.eye(n, order='F', dtype=float)
    r = np.eye(n, order='F', dtype=float)

    ferr, r_out, info = sb03sx('X', 'U', 'R', n, 1.0, t, u, r)

    assert info == -1


def test_sb03sx_invalid_uplo():
    """
    Test error handling for invalid UPLO parameter
    """
    n = 2
    t = np.eye(n, order='F', dtype=float)
    u = np.eye(n, order='F', dtype=float)
    r = np.eye(n, order='F', dtype=float)

    ferr, r_out, info = sb03sx('N', 'X', 'R', n, 1.0, t, u, r)

    assert info == -2


def test_sb03sx_invalid_lyapun():
    """
    Test error handling for invalid LYAPUN parameter
    """
    n = 2
    t = np.eye(n, order='F', dtype=float)
    u = np.eye(n, order='F', dtype=float)
    r = np.eye(n, order='F', dtype=float)

    ferr, r_out, info = sb03sx('N', 'U', 'X', n, 1.0, t, u, r)

    assert info == -3
