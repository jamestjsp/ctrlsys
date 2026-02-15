"""
Tests for MB02RD: Solve linear systems with Hessenberg LU factorization.

Uses numpy only.
"""

import numpy as np


def test_mb02rd_notrans():
    """
    Test MB02RD solving H*X = B (no transpose).

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb02sd, mb02rd

    np.random.seed(42)
    n = 4
    nrhs = 2

    h = np.zeros((n, n), order='F', dtype=float)
    for i in range(n):
        for j in range(n):
            if j >= i - 1:
                h[i, j] = np.random.randn()

    h[np.diag_indices(n)] += 2.0

    x_true = np.random.randn(n, nrhs).astype(float, order='F')
    b = h @ x_true

    h_orig = h.copy()
    h_lu, ipiv, info = mb02sd(n, h)
    assert info == 0

    b_copy = b.copy(order='F')
    x_out, info = mb02rd('N', n, nrhs, h_lu, ipiv, b_copy)

    assert info == 0
    np.testing.assert_allclose(x_out, x_true, rtol=1e-12, atol=1e-14)


def test_mb02rd_trans():
    """
    Test MB02RD solving H'*X = B (transpose).

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb02sd, mb02rd

    np.random.seed(123)
    n = 3
    nrhs = 1

    h = np.zeros((n, n), order='F', dtype=float)
    for i in range(n):
        for j in range(n):
            if j >= i - 1:
                h[i, j] = np.random.randn()

    h[np.diag_indices(n)] += 2.0

    x_true = np.random.randn(n, nrhs).astype(float, order='F')
    b = h.T @ x_true

    h_lu, ipiv, info = mb02sd(n, h)
    assert info == 0

    b_copy = b.copy(order='F')
    x_out, info = mb02rd('T', n, nrhs, h_lu, ipiv, b_copy)

    assert info == 0
    np.testing.assert_allclose(x_out, x_true, rtol=1e-12, atol=1e-14)


def test_mb02rd_n0():
    """Test MB02RD with n=0 (quick return)."""
    from ctrlsys import mb02sd, mb02rd

    n = 0
    nrhs = 1

    h = np.zeros((1, 1), order='F', dtype=float)
    b = np.zeros((1, 1), order='F', dtype=float)
    ipiv = np.zeros(1, dtype=np.int32)

    x_out, info = mb02rd('N', n, nrhs, h, ipiv, b)

    assert info == 0


def test_mb02rd_nrhs0():
    """Test MB02RD with nrhs=0 (quick return)."""
    from ctrlsys import mb02sd, mb02rd

    np.random.seed(456)
    n = 3
    nrhs = 0

    h = np.random.randn(n, n).astype(float, order='F')
    b = np.zeros((n, 1), order='F', dtype=float)
    ipiv = np.zeros(n, dtype=np.int32)

    x_out, info = mb02rd('N', n, nrhs, h, ipiv, b)

    assert info == 0


def test_mb02rd_invalid_trans():
    """Test MB02RD with invalid TRANS parameter."""
    from ctrlsys import mb02rd

    n = 2
    nrhs = 1

    h = np.eye(n, order='F', dtype=float)
    b = np.ones((n, nrhs), order='F', dtype=float)
    ipiv = np.array([1, 2], dtype=np.int32)

    x_out, info = mb02rd('X', n, nrhs, h, ipiv, b)

    assert info == -1
