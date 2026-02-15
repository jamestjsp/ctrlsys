"""
Tests for MB01XY: Compute U'*U or L*L' for triangular matrices.

MB01XY computes the matrix product U'*U or L*L', where U and L are
upper and lower triangular matrices, respectively, stored in the
corresponding triangular part of the array A.

If UPLO = 'U' then the upper triangle of U'*U is stored (overwrites U).
If UPLO = 'L' then the lower triangle of L*L' is stored (overwrites L).

This is a counterpart of LAPACK DLAUU2 which computes U*U' or L'*L.
"""
import numpy as np
import pytest
from ctrlsys import mb01xy


def test_mb01xy_upper_basic():
    """
    Test upper triangular case: U'*U for a simple 3x3 matrix.

    For U = [[1, 2, 3],
             [0, 4, 5],
             [0, 0, 6]]
    U'*U = [[1, 2, 3],          [[1, 0, 0],
            [2, 20, 23],  from   [2, 4, 0],
            [3, 23, 70]]         [3, 5, 6]]

    We verify: (U'*U)[i,j] = sum_k U[k,i]*U[k,j] for i<=j (upper part)
    """
    u = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 4.0, 5.0],
        [0.0, 0.0, 6.0]
    ], order='F', dtype=float)

    expected_upper = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 20.0, 26.0],
        [0.0, 0.0, 70.0]
    ], order='F', dtype=float)

    result, info = mb01xy('U', u.copy(order='F'))

    assert info == 0
    np.testing.assert_allclose(np.triu(result), expected_upper, rtol=1e-14)


def test_mb01xy_lower_basic():
    """
    Test lower triangular case: L*L' for a simple 3x3 matrix.

    For L = [[1, 0, 0],
             [2, 4, 0],
             [3, 5, 6]]
    L*L' = [[1, 2, 3],           [[1, 0, 0],
            [2, 20, 26], from     [2, 4, 0],
            [3, 26, 70]]          [3, 5, 6]] * L'

    We verify: (L*L')[i,j] = sum_k L[i,k]*L[j,k] for i>=j (lower part)
    """
    l = np.array([
        [1.0, 0.0, 0.0],
        [2.0, 4.0, 0.0],
        [3.0, 5.0, 6.0]
    ], order='F', dtype=float)

    expected_lower = np.array([
        [1.0, 0.0, 0.0],
        [2.0, 20.0, 0.0],
        [3.0, 26.0, 70.0]
    ], order='F', dtype=float)

    result, info = mb01xy('L', l.copy(order='F'))

    assert info == 0
    np.testing.assert_allclose(np.tril(result), expected_lower, rtol=1e-14)


def test_mb01xy_upper_property_symmetric():
    """
    Property test: U'*U is symmetric positive semi-definite.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 4
    u = np.triu(np.random.randn(n, n)).astype(float, order='F')

    result, info = mb01xy('U', u.copy(order='F'))

    assert info == 0

    full_result = np.triu(result) + np.triu(result, 1).T
    np.testing.assert_allclose(full_result, full_result.T, rtol=1e-14)

    eigenvalues = np.linalg.eigvalsh(full_result)
    assert np.all(eigenvalues >= -1e-14)


def test_mb01xy_lower_property_symmetric():
    """
    Property test: L*L' is symmetric positive semi-definite.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 4
    l = np.tril(np.random.randn(n, n)).astype(float, order='F')

    result, info = mb01xy('L', l.copy(order='F'))

    assert info == 0

    full_result = np.tril(result) + np.tril(result, -1).T
    np.testing.assert_allclose(full_result, full_result.T, rtol=1e-14)

    eigenvalues = np.linalg.eigvalsh(full_result)
    assert np.all(eigenvalues >= -1e-14)


def test_mb01xy_upper_vs_numpy():
    """
    Validate upper case: U'*U matches numpy U.T @ U (upper triangle only).

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 5
    u = np.triu(np.random.randn(n, n)).astype(float, order='F')
    u_copy = u.copy(order='F')

    result, info = mb01xy('U', u_copy)

    assert info == 0

    expected_full = u.T @ u
    np.testing.assert_allclose(np.triu(result), np.triu(expected_full), rtol=1e-14)


def test_mb01xy_lower_vs_numpy():
    """
    Validate lower case: L*L' matches numpy L @ L.T (lower triangle only).

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 5
    l = np.tril(np.random.randn(n, n)).astype(float, order='F')
    l_copy = l.copy(order='F')

    result, info = mb01xy('L', l_copy)

    assert info == 0

    expected_full = l @ l.T
    np.testing.assert_allclose(np.tril(result), np.tril(expected_full), rtol=1e-14)


def test_mb01xy_n1():
    """
    Test with N=1 (scalar case): a'*a = a*a = a^2.
    """
    u = np.array([[3.0]], order='F', dtype=float)

    result, info = mb01xy('U', u.copy(order='F'))

    assert info == 0
    np.testing.assert_allclose(result, np.array([[9.0]]), rtol=1e-14)

    l = np.array([[4.0]], order='F', dtype=float)

    result, info = mb01xy('L', l.copy(order='F'))

    assert info == 0
    np.testing.assert_allclose(result, np.array([[16.0]]), rtol=1e-14)


def test_mb01xy_n0():
    """
    Test with N=0 (empty matrix).
    """
    a = np.zeros((0, 0), order='F', dtype=float)

    result, info = mb01xy('U', a.copy(order='F'))
    assert info == 0

    result, info = mb01xy('L', a.copy(order='F'))
    assert info == 0


def test_mb01xy_identity():
    """
    Property test: I'*I = I (identity preserved).
    """
    n = 4
    i_matrix = np.eye(n, order='F', dtype=float)

    result_u, info_u = mb01xy('U', i_matrix.copy(order='F'))
    assert info_u == 0
    np.testing.assert_allclose(np.triu(result_u), np.triu(i_matrix), rtol=1e-14)

    result_l, info_l = mb01xy('L', i_matrix.copy(order='F'))
    assert info_l == 0
    np.testing.assert_allclose(np.tril(result_l), np.tril(i_matrix), rtol=1e-14)


def test_mb01xy_cholesky_relation():
    """
    Property test: If A = L*L' (Cholesky), then mb01xy(L,'L') gives A.

    Random seed: 888 (for reproducibility)
    """
    np.random.seed(888)
    n = 4
    tmp = np.random.randn(n, n)
    a = tmp @ tmp.T + np.eye(n)
    l = np.linalg.cholesky(a)

    l_f = np.asarray(l, order='F', dtype=float)
    result, info = mb01xy('L', l_f.copy(order='F'))

    assert info == 0
    np.testing.assert_allclose(np.tril(result), np.tril(a), rtol=1e-13)


def test_mb01xy_invalid_uplo():
    """
    Test error handling for invalid UPLO parameter.
    """
    a = np.array([[1.0, 2.0], [0.0, 3.0]], order='F', dtype=float)

    result, info = mb01xy('X', a.copy(order='F'))
    assert info == -1


def test_mb01xy_invalid_n():
    """
    Test error handling for negative N.

    Note: This depends on wrapper implementation handling negative dimensions.
    """
    pass


def test_mb01xy_large_matrix():
    """
    Test with larger matrix for numerical stability.

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    n = 20
    u = np.triu(np.random.randn(n, n)).astype(float, order='F')
    u_copy = u.copy(order='F')

    result, info = mb01xy('U', u_copy)

    assert info == 0

    expected_full = u.T @ u
    np.testing.assert_allclose(np.triu(result), np.triu(expected_full), rtol=1e-12)
