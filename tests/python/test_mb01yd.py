"""
Tests for MB01YD: Symmetric rank k operation with banded matrix.

MB01YD performs the symmetric rank k operations:
    C := alpha*op(A)*op(A)' + beta*C
where alpha and beta are scalars, C is an n-by-n symmetric matrix,
op(A) is an n-by-k matrix, and op(A) is A or A'.

The matrix A has l nonzero codiagonals, either upper or lower:
- If UPLO = 'U': A has L nonzero subdiagonals (upper triang + L subdiag)
- If UPLO = 'L': A has L nonzero superdiagonals (lower triang + L superdiag)

This is a specialization of DSYRK for banded matrices.
"""
import numpy as np
import pytest
from ctrlsys import mb01yd


def test_mb01yd_upper_notrans_basic():
    """
    Test UPLO='U', TRANS='N': C := alpha*A*A' + beta*C (upper triangle).

    For a simple 3x3 case with n=3, k=3, l=1 (Hessenberg-like).
    A has upper triangular + 1 subdiagonal.
    """
    n, k, l = 3, 3, 1
    alpha, beta = 1.0, 0.0

    a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [0.0, 7.0, 8.0]
    ], order='F', dtype=float)

    c = np.zeros((n, n), order='F', dtype=float)

    c_result, info = mb01yd('U', 'N', n, k, l, alpha, beta, a, c)

    assert info == 0

    full_a = a.copy()
    full_a[2, 0] = 0.0
    expected_full = full_a @ full_a.T
    np.testing.assert_allclose(np.triu(c_result), np.triu(expected_full), rtol=1e-14)


def test_mb01yd_lower_notrans_basic():
    """
    Test UPLO='L', TRANS='N': C := alpha*A*A' + beta*C (lower triangle).

    For a simple 3x3 case with n=3, k=3, l=1 (lower triang + 1 superdiag).
    """
    n, k, l = 3, 3, 1
    alpha, beta = 1.0, 0.0

    a = np.array([
        [1.0, 4.0, 0.0],
        [2.0, 5.0, 7.0],
        [3.0, 6.0, 8.0]
    ], order='F', dtype=float)

    c = np.zeros((n, n), order='F', dtype=float)

    c_result, info = mb01yd('L', 'N', n, k, l, alpha, beta, a, c)

    assert info == 0

    full_a = a.copy()
    full_a[0, 2] = 0.0
    expected_full = full_a @ full_a.T
    np.testing.assert_allclose(np.tril(c_result), np.tril(expected_full), rtol=1e-14)


def test_mb01yd_upper_trans_basic():
    """
    Test UPLO='U', TRANS='T': C := alpha*A'*A + beta*C (upper triangle).

    For a k=3, n=4 case with l=1.
    """
    k, n, l = 3, 4, 1
    alpha, beta = 1.0, 0.0

    a = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [0.0, 9.0, 10.0, 11.0]
    ], order='F', dtype=float)

    c = np.zeros((n, n), order='F', dtype=float)

    c_result, info = mb01yd('U', 'T', n, k, l, alpha, beta, a, c)

    assert info == 0

    full_a = a.copy()
    full_a[2, 0] = 0.0
    expected_full = full_a.T @ full_a
    np.testing.assert_allclose(np.triu(c_result), np.triu(expected_full), rtol=1e-14)


def test_mb01yd_lower_trans_basic():
    """
    Test UPLO='L', TRANS='T': C := alpha*A'*A + beta*C (lower triangle).

    For a k=3, n=4 case with l=1.
    """
    k, n, l = 3, 4, 1
    alpha, beta = 1.0, 0.0

    a = np.array([
        [1.0, 5.0, 0.0, 0.0],
        [2.0, 6.0, 9.0, 0.0],
        [3.0, 7.0, 10.0, 12.0]
    ], order='F', dtype=float)

    c = np.zeros((n, n), order='F', dtype=float)

    c_result, info = mb01yd('L', 'T', n, k, l, alpha, beta, a, c)

    assert info == 0

    full_a = a.copy()
    full_a[0, 2] = 0.0
    full_a[0, 3] = 0.0
    full_a[1, 3] = 0.0
    expected_full = full_a.T @ full_a
    np.testing.assert_allclose(np.tril(c_result), np.tril(expected_full), rtol=1e-14)


def test_mb01yd_scaling_alpha_beta():
    """
    Test alpha and beta scaling: C := alpha*A*A' + beta*C.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n, k, l = 4, 4, 1
    alpha, beta = 2.5, 0.5

    a = np.triu(np.random.randn(n, k), -l).astype(float, order='F')

    c = np.zeros((n, n), order='F', dtype=float)
    c_init = np.random.randn(n, n)
    c_init = (c_init + c_init.T) / 2
    np.copyto(c, np.triu(c_init), where=np.triu(np.ones((n, n), dtype=bool)))
    c = np.asarray(c, order='F')

    c_copy = c.copy(order='F')

    c_result, info = mb01yd('U', 'N', n, k, l, alpha, beta, a, c_copy)

    assert info == 0

    mask = np.triu(np.ones((n, n), dtype=bool))
    for i in range(l + 1, n):
        for j in range(i - l):
            a[i, j] = 0.0
    expected = alpha * (a @ a.T) + beta * c
    np.testing.assert_allclose(c_result[mask], expected[mask], rtol=1e-13)


def test_mb01yd_result_symmetric():
    """
    Property test: Result C should be symmetric.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n, k, l = 5, 5, 2
    alpha, beta = 1.0, 0.0

    a = np.triu(np.random.randn(n, k), -l).astype(float, order='F')
    c = np.zeros((n, n), order='F', dtype=float)

    c_upper, info = mb01yd('U', 'N', n, k, l, alpha, beta, a, c.copy(order='F'))
    assert info == 0

    a_lower = np.tril(np.random.randn(n, k), l).astype(float, order='F')
    c_lower, info = mb01yd('L', 'N', n, k, l, alpha, beta, a_lower, c.copy(order='F'))
    assert info == 0

    full_upper = np.triu(c_upper) + np.triu(c_upper, 1).T
    np.testing.assert_allclose(full_upper, full_upper.T, rtol=1e-14)

    full_lower = np.tril(c_lower) + np.tril(c_lower, -1).T
    np.testing.assert_allclose(full_lower, full_lower.T, rtol=1e-14)


def test_mb01yd_alpha_zero():
    """
    Test alpha=0: C := beta*C (A not referenced).
    """
    n, k, l = 3, 3, 1
    alpha, beta = 0.0, 2.0

    a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [0.0, 7.0, 8.0]
    ], order='F', dtype=float)

    c = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 4.0, 5.0],
        [0.0, 0.0, 6.0]
    ], order='F', dtype=float)

    c_copy = c.copy(order='F')
    c_result, info = mb01yd('U', 'N', n, k, l, alpha, beta, a, c_copy)

    assert info == 0
    expected = beta * c
    np.testing.assert_allclose(np.triu(c_result), np.triu(expected), rtol=1e-14)


def test_mb01yd_beta_zero():
    """
    Test beta=0: C := alpha*A*A' (C need not be set).
    """
    n, k, l = 3, 3, 1
    alpha, beta = 1.0, 0.0

    a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [0.0, 7.0, 8.0]
    ], order='F', dtype=float)

    c = np.full((n, n), np.nan, order='F', dtype=float)

    c_result, info = mb01yd('U', 'N', n, k, l, alpha, beta, a, c)

    assert info == 0

    full_a = a.copy()
    full_a[2, 0] = 0.0
    expected = alpha * (full_a @ full_a.T)
    np.testing.assert_allclose(np.triu(c_result), np.triu(expected), rtol=1e-14)


def test_mb01yd_alpha_beta_zero():
    """
    Test alpha=0, beta=0: C := 0.
    """
    n, k, l = 3, 3, 1
    alpha, beta = 0.0, 0.0

    a = np.random.randn(n, k).astype(float, order='F')
    c = np.random.randn(n, n).astype(float, order='F')

    c_copy = c.copy(order='F')
    c_result, info = mb01yd('U', 'N', n, k, l, alpha, beta, a, c_copy)

    assert info == 0
    np.testing.assert_allclose(np.triu(c_result), np.zeros((n, n)), rtol=1e-14)


def test_mb01yd_l0_triangular():
    """
    Test l=0: A is upper/lower triangular (no extra diagonals).

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n, k, l = 4, 4, 0
    alpha, beta = 1.0, 0.0

    a = np.triu(np.random.randn(n, k)).astype(float, order='F')
    c = np.zeros((n, n), order='F', dtype=float)

    c_result, info = mb01yd('U', 'N', n, k, l, alpha, beta, a, c)

    assert info == 0
    expected = a @ a.T
    np.testing.assert_allclose(np.triu(c_result), np.triu(expected), rtol=1e-14)


def test_mb01yd_n0():
    """
    Test n=0 (empty case).
    """
    n, k, l = 0, 0, 0
    alpha, beta = 1.0, 1.0

    a = np.zeros((0, 0), order='F', dtype=float)
    c = np.zeros((0, 0), order='F', dtype=float)

    c_result, info = mb01yd('U', 'N', n, k, l, alpha, beta, a, c)

    assert info == 0


def test_mb01yd_k0():
    """
    Test k=0: op(A) has no columns, so A*A' = 0.
    """
    n, k, l = 3, 0, 0
    alpha, beta = 1.0, 2.0

    a = np.zeros((n, 0), order='F', dtype=float)
    c = np.eye(n, order='F', dtype=float)

    c_copy = c.copy(order='F')
    c_result, info = mb01yd('U', 'N', n, k, l, alpha, beta, a, c_copy)

    assert info == 0
    expected = beta * np.eye(n)
    np.testing.assert_allclose(np.triu(c_result), np.triu(expected), rtol=1e-14)


def test_mb01yd_invalid_uplo():
    """
    Test error handling for invalid UPLO parameter.
    """
    n, k, l = 3, 3, 1
    a = np.random.randn(n, k).astype(float, order='F')
    c = np.zeros((n, n), order='F', dtype=float)

    c_result, info = mb01yd('X', 'N', n, k, l, 1.0, 0.0, a, c)
    assert info == -1


def test_mb01yd_invalid_trans():
    """
    Test error handling for invalid TRANS parameter.
    """
    n, k, l = 3, 3, 1
    a = np.random.randn(n, k).astype(float, order='F')
    c = np.zeros((n, n), order='F', dtype=float)

    c_result, info = mb01yd('U', 'X', n, k, l, 1.0, 0.0, a, c)
    assert info == -2


def test_mb01yd_invalid_l():
    """
    Test error handling for invalid L parameter.

    L must satisfy 0 <= L <= max(0, M-1) where M depends on UPLO/TRANS.
    """
    n, k, l = 3, 3, 10
    a = np.random.randn(n, k).astype(float, order='F')
    c = np.zeros((n, n), order='F', dtype=float)

    c_result, info = mb01yd('U', 'N', n, k, l, 1.0, 0.0, a, c)
    assert info == -5


def test_mb01yd_positive_semidefinite():
    """
    Property test: A*A' is positive semi-definite.

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n, k, l = 5, 6, 2
    alpha, beta = 1.0, 0.0

    a = np.triu(np.random.randn(n, k), -l).astype(float, order='F')
    c = np.zeros((n, n), order='F', dtype=float)

    c_result, info = mb01yd('U', 'N', n, k, l, alpha, beta, a, c)

    assert info == 0

    full_c = np.triu(c_result) + np.triu(c_result, 1).T
    eigenvalues = np.linalg.eigvalsh(full_c)
    assert np.all(eigenvalues >= -1e-10)


def test_mb01yd_rectangular_notrans():
    """
    Test non-square A with TRANS='N': n != k.

    Random seed: 888 (for reproducibility)
    """
    np.random.seed(888)
    n, k, l = 5, 3, 1
    alpha, beta = 1.0, 0.0

    a = np.random.randn(n, k).astype(float, order='F')
    for i in range(l + 1, n):
        for j in range(min(k, i - l)):
            a[i, j] = 0.0

    c = np.zeros((n, n), order='F', dtype=float)

    c_result, info = mb01yd('U', 'N', n, k, l, alpha, beta, a, c)

    assert info == 0

    expected = a @ a.T
    np.testing.assert_allclose(np.triu(c_result), np.triu(expected), rtol=1e-13)


def test_mb01yd_rectangular_trans():
    """
    Test non-square A with TRANS='T': k != n.

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    k, n, l = 3, 5, 1
    alpha, beta = 1.0, 0.0

    a = np.random.randn(k, n).astype(float, order='F')
    for i in range(l + 1, k):
        for j in range(min(n, i - l)):
            a[i, j] = 0.0

    c = np.zeros((n, n), order='F', dtype=float)

    c_result, info = mb01yd('U', 'T', n, k, l, alpha, beta, a, c)

    assert info == 0

    expected = a.T @ a
    np.testing.assert_allclose(np.triu(c_result), np.triu(expected), rtol=1e-13)
