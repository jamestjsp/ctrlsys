"""
Tests for MB01XD: Compute U'*U or L*L' for triangular matrices (block algorithm).

MB01XD computes the matrix product U'*U or L*L', where U and L are
upper and lower triangular matrices, respectively, stored in the
corresponding triangular part of the array A.

If UPLO = 'U' then the upper triangle of U'*U is stored (overwrites U).
If UPLO = 'L' then the lower triangle of L*L' is stored (overwrites L).

This is a block algorithm counterpart of MB01XY. Uses BLAS 3 operations
when matrix size exceeds block size, otherwise calls MB01XY directly.
"""
import numpy as np
import pytest
from ctrlsys import mb01xd


def test_mb01xd_upper_basic():
    """
    Test upper triangular case: U'*U for a simple 3x3 matrix.

    For U = [[1, 2, 3],
             [0, 4, 5],
             [0, 0, 6]]
    U'*U = [[1, 2, 3],
            [2, 20, 26],
            [3, 26, 70]]

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

    result, info = mb01xd('U', u.copy(order='F'))

    assert info == 0
    np.testing.assert_allclose(np.triu(result), expected_upper, rtol=1e-14)


def test_mb01xd_lower_basic():
    """
    Test lower triangular case: L*L' for a simple 3x3 matrix.

    For L = [[1, 0, 0],
             [2, 4, 0],
             [3, 5, 6]]
    L*L' = [[1, 2, 3],
            [2, 20, 26],
            [3, 26, 70]]

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

    result, info = mb01xd('L', l.copy(order='F'))

    assert info == 0
    np.testing.assert_allclose(np.tril(result), expected_lower, rtol=1e-14)


def test_mb01xd_upper_property_symmetric():
    """
    Property test: U'*U is symmetric positive semi-definite.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 4
    u = np.triu(np.random.randn(n, n)).astype(float, order='F')

    result, info = mb01xd('U', u.copy(order='F'))

    assert info == 0

    full_result = np.triu(result) + np.triu(result, 1).T
    np.testing.assert_allclose(full_result, full_result.T, rtol=1e-14)

    eigenvalues = np.linalg.eigvalsh(full_result)
    assert np.all(eigenvalues >= -1e-14)


def test_mb01xd_lower_property_symmetric():
    """
    Property test: L*L' is symmetric positive semi-definite.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 4
    l = np.tril(np.random.randn(n, n)).astype(float, order='F')

    result, info = mb01xd('L', l.copy(order='F'))

    assert info == 0

    full_result = np.tril(result) + np.tril(result, -1).T
    np.testing.assert_allclose(full_result, full_result.T, rtol=1e-14)

    eigenvalues = np.linalg.eigvalsh(full_result)
    assert np.all(eigenvalues >= -1e-14)


def test_mb01xd_upper_vs_numpy():
    """
    Validate upper case: U'*U matches numpy U.T @ U (upper triangle only).

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 5
    u = np.triu(np.random.randn(n, n)).astype(float, order='F')
    u_copy = u.copy(order='F')

    result, info = mb01xd('U', u_copy)

    assert info == 0

    expected_full = u.T @ u
    np.testing.assert_allclose(np.triu(result), np.triu(expected_full), rtol=1e-14)


def test_mb01xd_lower_vs_numpy():
    """
    Validate lower case: L*L' matches numpy L @ L.T (lower triangle only).

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 5
    l = np.tril(np.random.randn(n, n)).astype(float, order='F')
    l_copy = l.copy(order='F')

    result, info = mb01xd('L', l_copy)

    assert info == 0

    expected_full = l @ l.T
    np.testing.assert_allclose(np.tril(result), np.tril(expected_full), rtol=1e-14)


def test_mb01xd_n1():
    """
    Test with N=1 (scalar case): a'*a = a*a = a^2.
    """
    u = np.array([[3.0]], order='F', dtype=float)

    result, info = mb01xd('U', u.copy(order='F'))

    assert info == 0
    np.testing.assert_allclose(result, np.array([[9.0]]), rtol=1e-14)

    l = np.array([[4.0]], order='F', dtype=float)

    result, info = mb01xd('L', l.copy(order='F'))

    assert info == 0
    np.testing.assert_allclose(result, np.array([[16.0]]), rtol=1e-14)


def test_mb01xd_n0():
    """
    Test with N=0 (empty matrix).
    """
    a = np.zeros((0, 0), order='F', dtype=float)

    result, info = mb01xd('U', a.copy(order='F'))
    assert info == 0

    result, info = mb01xd('L', a.copy(order='F'))
    assert info == 0


def test_mb01xd_identity():
    """
    Property test: I'*I = I (identity preserved).
    """
    n = 4
    i_matrix = np.eye(n, order='F', dtype=float)

    result_u, info_u = mb01xd('U', i_matrix.copy(order='F'))
    assert info_u == 0
    np.testing.assert_allclose(np.triu(result_u), np.triu(i_matrix), rtol=1e-14)

    result_l, info_l = mb01xd('L', i_matrix.copy(order='F'))
    assert info_l == 0
    np.testing.assert_allclose(np.tril(result_l), np.tril(i_matrix), rtol=1e-14)


def test_mb01xd_cholesky_relation():
    """
    Property test: If A = L*L' (Cholesky), then mb01xd(L,'L') gives A.

    Random seed: 888 (for reproducibility)
    """
    np.random.seed(888)
    n = 4
    tmp = np.random.randn(n, n)
    a = tmp @ tmp.T + np.eye(n)
    l = np.linalg.cholesky(a)

    l_f = np.asarray(l, order='F', dtype=float)
    result, info = mb01xd('L', l_f.copy(order='F'))

    assert info == 0
    np.testing.assert_allclose(np.tril(result), np.tril(a), rtol=1e-13)


def test_mb01xd_invalid_uplo():
    """
    Test error handling for invalid UPLO parameter.
    """
    a = np.array([[1.0, 2.0], [0.0, 3.0]], order='F', dtype=float)

    result, info = mb01xd('X', a.copy(order='F'))
    assert info == -1


def test_mb01xd_large_matrix_block_algorithm():
    """
    Test with larger matrix to exercise block algorithm (BLAS 3 path).

    When N > NB (block size, typically 64), the block algorithm is used.

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    n = 100
    u = np.triu(np.random.randn(n, n)).astype(float, order='F')
    u_copy = u.copy(order='F')

    result, info = mb01xd('U', u_copy)

    assert info == 0

    expected_full = u.T @ u
    np.testing.assert_allclose(np.triu(result), np.triu(expected_full), rtol=2e-12)


def test_mb01xd_large_matrix_lower():
    """
    Test lower triangular with larger matrix (block algorithm path).

    Random seed: 1234 (for reproducibility)
    """
    np.random.seed(1234)
    n = 100
    l = np.tril(np.random.randn(n, n)).astype(float, order='F')
    l_copy = l.copy(order='F')

    result, info = mb01xd('L', l_copy)

    assert info == 0

    expected_full = l @ l.T
    np.testing.assert_allclose(np.tril(result), np.tril(expected_full), rtol=2e-11)


def test_mb01xd_matches_mb01xy():
    """
    Verify MB01XD and MB01XY give same results for small matrices.

    Since MB01XD calls MB01XY for small matrices or as base case,
    results should be identical.

    Random seed: 555 (for reproducibility)
    """
    from ctrlsys import mb01xy

    np.random.seed(555)
    n = 5
    u = np.triu(np.random.randn(n, n)).astype(float, order='F')

    result_xd, info_xd = mb01xd('U', u.copy(order='F'))
    result_xy, info_xy = mb01xy('U', u.copy(order='F'))

    assert info_xd == 0
    assert info_xy == 0
    np.testing.assert_allclose(np.triu(result_xd), np.triu(result_xy), rtol=1e-14)

    l = np.tril(np.random.randn(n, n)).astype(float, order='F')

    result_xd, info_xd = mb01xd('L', l.copy(order='F'))
    result_xy, info_xy = mb01xy('L', l.copy(order='F'))

    assert info_xd == 0
    assert info_xy == 0
    np.testing.assert_allclose(np.tril(result_xd), np.tril(result_xy), rtol=1e-14)
