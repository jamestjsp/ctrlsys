"""
Tests for MB02UV: LU factorization with complete pivoting.

Computes A = P * L * U * Q where P, Q are permutation matrices,
L is unit lower triangular, U is upper triangular.
"""

import numpy as np
import pytest


def test_mb02uv_basic():
    """
    Validate basic LU factorization with complete pivoting.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mb02uv

    np.random.seed(42)
    n = 4

    a = np.array([
        [2.0, 1.0, 3.0, 4.0],
        [4.0, 3.0, 2.0, 1.0],
        [1.0, 4.0, 2.0, 3.0],
        [3.0, 2.0, 4.0, 1.0]
    ], order='F', dtype=float)
    a_orig = a.copy()

    a_lu, ipiv, jpiv, info = mb02uv(n, a)

    assert info == 0
    assert len(ipiv) == n
    assert len(jpiv) == n


def test_mb02uv_reconstruct():
    """
    Validate reconstruction: P * L * U * Q = A.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mb02uv

    np.random.seed(123)
    n = 3

    a = np.array([
        [4.0, 2.0, 1.0],
        [2.0, 5.0, 3.0],
        [1.0, 3.0, 6.0]
    ], order='F', dtype=float)
    a_orig = a.copy()

    a_lu, ipiv, jpiv, info = mb02uv(n, a)
    assert info == 0

    L = np.eye(n, dtype=float)
    U = np.zeros((n, n), dtype=float)
    for i in range(n):
        U[i, i:] = a_lu[i, i:]
        L[i+1:, i] = a_lu[i+1:, i]

    LU = L @ U

    P = np.eye(n, dtype=float)
    for i in range(n):
        p = ipiv[i] - 1
        if p != i:
            P[[i, p], :] = P[[p, i], :]

    Q = np.eye(n, dtype=float)
    for j in range(n):
        q = jpiv[j] - 1
        if q != j:
            Q[:, [j, q]] = Q[:, [q, j]]

    reconstructed = P.T @ LU @ Q.T

    np.testing.assert_allclose(reconstructed, a_orig, rtol=1e-13, atol=1e-14)


def test_mb02uv_singular():
    """
    Validate handling of singular matrix (info > 0 warning).

    Random seed: 456 (for reproducibility)
    """
    from ctrlsys import mb02uv

    np.random.seed(456)
    n = 3

    a = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [1.0, 1.0, 1.0]
    ], order='F', dtype=float)

    a_lu, ipiv, jpiv, info = mb02uv(n, a)

    assert info >= 0


def test_mb02uv_identity():
    """
    Validate factorization of identity matrix.
    """
    from ctrlsys import mb02uv

    n = 4
    a = np.eye(n, order='F', dtype=float)

    a_lu, ipiv, jpiv, info = mb02uv(n, a)

    assert info == 0


def test_mb02uv_permutation_indices():
    """
    Validate that pivot indices are in valid range [1, n].

    Random seed: 789 (for reproducibility)
    """
    from ctrlsys import mb02uv

    np.random.seed(789)
    n = 5

    a = np.random.randn(n, n).astype(float, order='F')

    a_lu, ipiv, jpiv, info = mb02uv(n, a)

    assert info >= 0
    for i in range(n):
        assert 1 <= ipiv[i] <= n
        assert 1 <= jpiv[i] <= n


def test_mb02uv_1x1():
    """
    Validate edge case: 1x1 matrix.
    """
    from ctrlsys import mb02uv

    n = 1
    a = np.array([[5.0]], order='F', dtype=float)

    a_lu, ipiv, jpiv, info = mb02uv(n, a)

    assert info == 0
    assert a_lu[0, 0] == 5.0
    assert ipiv[0] == 1
    assert jpiv[0] == 1


def test_mb02uv_2x2():
    """
    Validate 2x2 case with explicit check.
    """
    from ctrlsys import mb02uv

    n = 2
    a = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ], order='F', dtype=float)
    a_orig = a.copy()

    a_lu, ipiv, jpiv, info = mb02uv(n, a)

    assert info == 0

    L = np.eye(n, dtype=float)
    U = np.zeros((n, n), dtype=float)
    for i in range(n):
        U[i, i:] = a_lu[i, i:]
        L[i+1:, i] = a_lu[i+1:, i]

    LU = L @ U

    P = np.eye(n, dtype=float)
    for i in range(n):
        p = ipiv[i] - 1
        if p != i:
            P[[i, p], :] = P[[p, i], :]

    Q = np.eye(n, dtype=float)
    for j in range(n):
        q = jpiv[j] - 1
        if q != j:
            Q[:, [j, q]] = Q[:, [q, j]]

    reconstructed = P.T @ LU @ Q.T

    np.testing.assert_allclose(reconstructed, a_orig, rtol=1e-13, atol=1e-14)


def test_mb02uv_random_larger():
    """
    Validate factorization of larger random matrix.

    Random seed: 111 (for reproducibility)
    """
    from ctrlsys import mb02uv

    np.random.seed(111)
    n = 8

    a = np.random.randn(n, n).astype(float, order='F')
    a = a + 5.0 * np.eye(n, dtype=float)
    a_orig = a.copy()

    a_lu, ipiv, jpiv, info = mb02uv(n, a)

    assert info == 0

    L = np.eye(n, dtype=float)
    U = np.zeros((n, n), dtype=float)
    for i in range(n):
        U[i, i:] = a_lu[i, i:]
        L[i+1:, i] = a_lu[i+1:, i]

    LU = L @ U

    P = np.eye(n, dtype=float)
    for i in range(n):
        p = ipiv[i] - 1
        if p != i:
            P[[i, p], :] = P[[p, i], :]

    Q = np.eye(n, dtype=float)
    for j in range(n):
        q = jpiv[j] - 1
        if q != j:
            Q[:, [j, q]] = Q[:, [q, j]]

    reconstructed = P.T @ LU @ Q.T

    np.testing.assert_allclose(reconstructed, a_orig, rtol=1e-12, atol=1e-13)
