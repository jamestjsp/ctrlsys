"""
Tests for MB04ND - RQ factorization of special structured block matrix.

MB04ND calculates an RQ factorization of the first block row and applies
the orthogonal transformations (from the right) to the second block row:

    [ A   R ]        [ 0   R_new ]
    [       ] * Q' = [           ]
    [ C   B ]        [ C_new B_new ]

where R and R_new are upper triangular. Matrix A can be full (UPLO='F')
or upper trapezoidal/triangular (UPLO='U').

The routine uses N Householder transformations exploiting zero pattern.
"""

import numpy as np
import pytest


def compute_rq_reference(a, r, c, b, uplo='F'):
    """
    Reference implementation using NumPy QR.

    Computes RQ factorization of [A R] from the right side, then applies
    the same transformation to [C B].

    Returns:
        r_new: Updated upper triangular R
        a_new: Contains Householder vectors (first row annihilated)
        b_new: Updated B
        c_new: Updated C
    """
    n = r.shape[0]
    p = a.shape[1]
    m = b.shape[0]

    first_row = np.hstack([a, r])
    second_row = np.hstack([c, b])

    q, r_full = np.linalg.qr(first_row.T, mode='complete')

    r_new = r_full.T

    r_new_upper = np.zeros((n, n), dtype=float, order='F')
    for i in range(n):
        for j in range(i, n):
            r_new_upper[i, j] = r_new[i, p + j]

    second_transformed = second_row @ q

    c_new = second_transformed[:, :p].copy(order='F')
    b_new = second_transformed[:, p:].copy(order='F')

    return r_new_upper, c_new, b_new


def test_mb04nd_basic_full():
    """
    Basic test with UPLO='F' (full matrix A).

    Tests RQ factorization of small block matrix.
    Random seed: 42
    """
    np.random.seed(42)

    n, m, p = 3, 2, 3

    a = np.random.randn(n, p).astype(float, order='F')
    r = np.triu(np.random.randn(n, n)).astype(float, order='F')
    c = np.random.randn(m, p).astype(float, order='F')
    b = np.random.randn(m, n).astype(float, order='F')

    a_work = a.copy(order='F')
    r_work = r.copy(order='F')
    c_work = c.copy(order='F')
    b_work = b.copy(order='F')

    from ctrlsys import _slicot

    tau = _ctrlsys.mb04nd('F', n, m, p, r_work, a_work, b_work, c_work)

    assert len(tau) == n
    assert np.all(np.isfinite(r_work))
    assert np.all(np.isfinite(b_work))
    assert np.all(np.isfinite(c_work))

    for i in range(n):
        for j in range(i):
            assert abs(r_work[i, j]) < 1e-10, f"R not upper triangular at ({i},{j})"


def test_mb04nd_upper_triangular():
    """
    Test with UPLO='U' (upper triangular A).

    Random seed: 123
    """
    np.random.seed(123)

    n, m, p = 3, 2, 3

    a = np.zeros((n, p), dtype=float, order='F')
    for i in range(n):
        for j in range(max(0, p - n + i), p):
            a[i, j] = np.random.randn()
    r = np.triu(np.random.randn(n, n)).astype(float, order='F')
    c = np.random.randn(m, p).astype(float, order='F')
    b = np.random.randn(m, n).astype(float, order='F')

    a_work = a.copy(order='F')
    r_work = r.copy(order='F')
    c_work = c.copy(order='F')
    b_work = b.copy(order='F')

    from ctrlsys import _slicot

    tau = _ctrlsys.mb04nd('U', n, m, p, r_work, a_work, b_work, c_work)

    assert len(tau) == n
    for i in range(n):
        for j in range(i):
            assert abs(r_work[i, j]) < 1e-10, f"R not upper triangular at ({i},{j})"


def test_mb04nd_orthogonality():
    """
    Mathematical property: Q from Householder is orthogonal.

    Reconstructs Q from tau and Householder vectors and verifies Q'Q = I.
    Random seed: 456
    """
    np.random.seed(456)

    n, m, p = 3, 0, 4

    a = np.random.randn(n, p).astype(float, order='F')
    r = np.triu(np.random.randn(n, n)).astype(float, order='F')
    c = np.empty((m, p), dtype=float, order='F')
    b = np.empty((m, n), dtype=float, order='F')

    a_work = a.copy(order='F')
    r_work = r.copy(order='F')
    c_work = c.copy(order='F')
    b_work = b.copy(order='F')

    from ctrlsys import _slicot

    tau = _ctrlsys.mb04nd('F', n, m, p, r_work, a_work, b_work, c_work)

    q = np.eye(n + p, dtype=float)
    for i in range(n - 1, -1, -1):
        u = np.zeros(p + 1)
        u[0] = 1.0
        u[1:] = a_work[i, :]

        h = np.eye(p + 1) - tau[i] * np.outer(u, u)
        h_full = np.eye(n + p, dtype=float)
        h_full[i, i] = h[0, 0]
        h_full[i, n:] = h[0, 1:]
        h_full[n:, i] = h[1:, 0]
        h_full[n:, n:] = h[1:, 1:]

        q = q @ h_full

    qtq = q.T @ q
    np.testing.assert_allclose(qtq, np.eye(n + p), rtol=1e-13, atol=1e-14)


def test_mb04nd_n_zero():
    """
    Edge case: n=0 (empty R matrix).

    Should return immediately without modification.
    """
    n, m, p = 0, 3, 2

    a = np.empty((n, p), dtype=float, order='F')
    r = np.empty((n, n), dtype=float, order='F')
    c = np.random.randn(m, p).astype(float, order='F')
    b = np.empty((m, n), dtype=float, order='F')

    c_orig = c.copy()

    a_work = a.copy(order='F')
    r_work = r.copy(order='F')
    c_work = c.copy(order='F')
    b_work = b.copy(order='F')

    from ctrlsys import _slicot

    tau = _ctrlsys.mb04nd('F', n, m, p, r_work, a_work, b_work, c_work)

    assert len(tau) == 0
    np.testing.assert_array_equal(c_work, c_orig)


def test_mb04nd_p_zero():
    """
    Edge case: p=0 (empty A matrix).

    Should return immediately without modification.
    """
    n, m, p = 3, 2, 0

    a = np.empty((n, p), dtype=float, order='F')
    r = np.triu(np.random.randn(n, n)).astype(float, order='F')
    c = np.empty((m, p), dtype=float, order='F')
    b = np.random.randn(m, n).astype(float, order='F')

    r_orig = r.copy()
    b_orig = b.copy()

    a_work = a.copy(order='F')
    r_work = r.copy(order='F')
    c_work = c.copy(order='F')
    b_work = b.copy(order='F')

    from ctrlsys import _slicot

    tau = _ctrlsys.mb04nd('F', n, m, p, r_work, a_work, b_work, c_work)

    assert len(tau) == n
    np.testing.assert_array_equal(r_work, r_orig)
    np.testing.assert_array_equal(b_work, b_orig)


def test_mb04nd_m_zero():
    """
    Edge case: m=0 (empty B, C matrices).

    Only first block row is transformed.
    Random seed: 789
    """
    np.random.seed(789)

    n, m, p = 3, 0, 4

    a = np.random.randn(n, p).astype(float, order='F')
    r = np.triu(np.random.randn(n, n)).astype(float, order='F')
    c = np.empty((m, p), dtype=float, order='F')
    b = np.empty((m, n), dtype=float, order='F')

    a_work = a.copy(order='F')
    r_work = r.copy(order='F')
    c_work = c.copy(order='F')
    b_work = b.copy(order='F')

    from ctrlsys import _slicot

    tau = _ctrlsys.mb04nd('F', n, m, p, r_work, a_work, b_work, c_work)

    assert len(tau) == n
    for i in range(n):
        for j in range(i):
            assert abs(r_work[i, j]) < 1e-10


def test_mb04nd_large_matrix():
    """
    Test larger matrices to exercise general BLAS code paths.

    Random seed: 888
    """
    np.random.seed(888)

    n, m, p = 8, 5, 12

    a = np.random.randn(n, p).astype(float, order='F')
    r = np.triu(np.random.randn(n, n)).astype(float, order='F')
    c = np.random.randn(m, p).astype(float, order='F')
    b = np.random.randn(m, n).astype(float, order='F')

    a_work = a.copy(order='F')
    r_work = r.copy(order='F')
    c_work = c.copy(order='F')
    b_work = b.copy(order='F')

    from ctrlsys import _slicot

    tau = _ctrlsys.mb04nd('F', n, m, p, r_work, a_work, b_work, c_work)

    assert len(tau) == n
    for i in range(n):
        for j in range(i):
            assert abs(r_work[i, j]) < 1e-10


def test_mb04nd_n_greater_p():
    """
    Test case where N > P with UPLO='U'.

    Upper trapezoidal case.
    Random seed: 999
    """
    np.random.seed(999)

    n, m, p = 5, 3, 3

    a = np.zeros((n, p), dtype=float, order='F')
    for i in range(n):
        for j in range(max(0, i - (n - p)), p):
            a[i, j] = np.random.randn()

    r = np.triu(np.random.randn(n, n)).astype(float, order='F')
    c = np.random.randn(m, p).astype(float, order='F')
    b = np.random.randn(m, n).astype(float, order='F')

    a_work = a.copy(order='F')
    r_work = r.copy(order='F')
    c_work = c.copy(order='F')
    b_work = b.copy(order='F')

    from ctrlsys import _slicot

    tau = _ctrlsys.mb04nd('U', n, m, p, r_work, a_work, b_work, c_work)

    assert len(tau) == n
    for i in range(n):
        for j in range(i):
            assert abs(r_work[i, j]) < 1e-10
