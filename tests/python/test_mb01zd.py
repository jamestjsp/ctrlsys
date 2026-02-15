"""
Tests for MB01ZD: Hessenberg-triangular matrix product.

MB01ZD computes the matrix product:
    H := alpha*op(T)*H   (SIDE='L'), or
    H := alpha*H*op(T)   (SIDE='R'),
where alpha is a scalar, H is an m-by-n upper or lower Hessenberg-like
matrix (with L nonzero subdiagonals or superdiagonals), T is a triangular
matrix, and op(T) is T or T'.

Hessenberg-like pattern (m=7, n=6, l=2):
    UPLO = 'U'                    UPLO = 'L'
    [ x x x x x x ]               [ x x x 0 0 0 ]
    [ x x x x x x ]               [ x x x x 0 0 ]
    [ x x x x x x ]               [ x x x x x 0 ]
    [ 0 x x x x x ]    vs         [ x x x x x x ]
    [ 0 0 x x x x ]               [ x x x x x x ]
    [ 0 0 0 x x x ]               [ x x x x x x ]
    [ 0 0 0 0 x x ]               [ x x x x x x ]
"""
import numpy as np
import pytest
from ctrlsys import mb01zd


def test_mb01zd_left_upper_notrans():
    """
    Test SIDE='L', UPLO='U', TRANS='N': H := alpha*T*H.

    Upper triangular T, upper Hessenberg H (l=1 subdiagonals).
    """
    m, n, l = 4, 3, 1
    alpha = 2.0

    t = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [0.0, 5.0, 6.0, 7.0],
        [0.0, 0.0, 8.0, 9.0],
        [0.0, 0.0, 0.0, 10.0]
    ], order='F', dtype=float)

    h = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [0.0, 7.0, 8.0],
        [0.0, 0.0, 9.0]
    ], order='F', dtype=float)

    h_copy = h.copy(order='F')
    h_result, info = mb01zd('L', 'U', 'N', 'N', m, n, l, alpha, t, h_copy)

    assert info == 0

    expected = alpha * t @ h
    np.testing.assert_allclose(h_result, expected, rtol=1e-14)


def test_mb01zd_left_upper_trans():
    """
    Test SIDE='L', UPLO='U', TRANS='T': H := alpha*T'*H.

    Upper triangular T, upper Hessenberg H.
    Result may be full matrix (not Hessenberg).
    """
    m, n, l = 4, 3, 1
    alpha = 1.0

    t = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [0.0, 5.0, 6.0, 7.0],
        [0.0, 0.0, 8.0, 9.0],
        [0.0, 0.0, 0.0, 10.0]
    ], order='F', dtype=float)

    h = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [0.0, 7.0, 8.0],
        [0.0, 0.0, 9.0]
    ], order='F', dtype=float)

    h_copy = h.copy(order='F')
    h_result, info = mb01zd('L', 'U', 'T', 'N', m, n, l, alpha, t, h_copy)

    assert info == 0

    expected = alpha * t.T @ h
    np.testing.assert_allclose(h_result, expected, rtol=1e-14)


def test_mb01zd_right_upper_notrans():
    """
    Test SIDE='R', UPLO='U', TRANS='N': H := alpha*H*T.

    Upper triangular T (n x n), upper Hessenberg H (l=1 subdiag).
    """
    m, n, l = 4, 3, 1
    alpha = 1.5

    t = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 4.0, 5.0],
        [0.0, 0.0, 6.0]
    ], order='F', dtype=float)

    h = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [0.0, 7.0, 8.0],
        [0.0, 0.0, 9.0]
    ], order='F', dtype=float)

    h_copy = h.copy(order='F')
    h_result, info = mb01zd('R', 'U', 'N', 'N', m, n, l, alpha, t, h_copy)

    assert info == 0

    expected = alpha * h @ t
    np.testing.assert_allclose(h_result, expected, rtol=1e-14)


def test_mb01zd_right_upper_trans():
    """
    Test SIDE='R', UPLO='U', TRANS='T': H := alpha*H*T'.

    Result may be larger than original Hessenberg pattern.
    """
    m, n, l = 4, 3, 1
    alpha = 1.0

    t = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 4.0, 5.0],
        [0.0, 0.0, 6.0]
    ], order='F', dtype=float)

    h = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [0.0, 7.0, 8.0],
        [0.0, 0.0, 9.0]
    ], order='F', dtype=float)

    h_copy = h.copy(order='F')
    h_result, info = mb01zd('R', 'U', 'T', 'N', m, n, l, alpha, t, h_copy)

    assert info == 0

    expected = alpha * h @ t.T
    np.testing.assert_allclose(h_result, expected, rtol=1e-14)


def test_mb01zd_left_lower_notrans():
    """
    Test SIDE='L', UPLO='L', TRANS='N': H := alpha*T*H.

    Lower triangular T, lower Hessenberg H (l=1 superdiagonals).
    """
    m, n, l = 4, 3, 1
    alpha = 1.0

    t = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [2.0, 3.0, 0.0, 0.0],
        [4.0, 5.0, 6.0, 0.0],
        [7.0, 8.0, 9.0, 10.0]
    ], order='F', dtype=float)

    h = np.array([
        [1.0, 2.0, 0.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0]
    ], order='F', dtype=float)

    h_copy = h.copy(order='F')
    h_result, info = mb01zd('L', 'L', 'N', 'N', m, n, l, alpha, t, h_copy)

    assert info == 0

    expected = alpha * t @ h
    np.testing.assert_allclose(h_result, expected, rtol=1e-14)


def test_mb01zd_left_lower_trans():
    """
    Test SIDE='L', UPLO='L', TRANS='T': H := alpha*T'*H.

    Lower triangular T transposed, lower Hessenberg H.
    """
    m, n, l = 4, 3, 1
    alpha = 1.0

    t = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [2.0, 3.0, 0.0, 0.0],
        [4.0, 5.0, 6.0, 0.0],
        [7.0, 8.0, 9.0, 10.0]
    ], order='F', dtype=float)

    h = np.array([
        [1.0, 2.0, 0.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0]
    ], order='F', dtype=float)

    h_copy = h.copy(order='F')
    h_result, info = mb01zd('L', 'L', 'T', 'N', m, n, l, alpha, t, h_copy)

    assert info == 0

    expected = alpha * t.T @ h
    np.testing.assert_allclose(h_result, expected, rtol=1e-14)


def test_mb01zd_right_lower_notrans():
    """
    Test SIDE='R', UPLO='L', TRANS='N': H := alpha*H*T.

    Lower triangular T (n x n), lower Hessenberg H.
    """
    m, n, l = 4, 3, 1
    alpha = 2.0

    t = np.array([
        [1.0, 0.0, 0.0],
        [2.0, 3.0, 0.0],
        [4.0, 5.0, 6.0]
    ], order='F', dtype=float)

    h = np.array([
        [1.0, 2.0, 0.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0]
    ], order='F', dtype=float)

    h_copy = h.copy(order='F')
    h_result, info = mb01zd('R', 'L', 'N', 'N', m, n, l, alpha, t, h_copy)

    assert info == 0

    expected = alpha * h @ t
    np.testing.assert_allclose(h_result, expected, rtol=1e-14)


def test_mb01zd_unit_triangular():
    """
    Test DIAG='U': Unit triangular T (diagonal assumed to be 1).
    """
    m, n, l = 3, 3, 1
    alpha = 1.0

    t = np.array([
        [999.0, 2.0, 3.0],
        [0.0, 999.0, 4.0],
        [0.0, 0.0, 999.0]
    ], order='F', dtype=float)

    h = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [0.0, 7.0, 8.0]
    ], order='F', dtype=float)

    h_copy = h.copy(order='F')
    h_result, info = mb01zd('L', 'U', 'N', 'U', m, n, l, alpha, t, h_copy)

    assert info == 0

    t_unit = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 1.0, 4.0],
        [0.0, 0.0, 1.0]
    ], order='F', dtype=float)
    expected = alpha * t_unit @ h
    np.testing.assert_allclose(h_result, expected, rtol=1e-14)


def test_mb01zd_alpha_zero():
    """
    Test alpha=0: H set to zero (T not referenced).
    """
    m, n, l = 3, 3, 1
    alpha = 0.0

    t = np.full((m, m), np.nan, order='F', dtype=float)

    h = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [0.0, 7.0, 8.0]
    ], order='F', dtype=float)

    h_copy = h.copy(order='F')
    h_result, info = mb01zd('L', 'U', 'N', 'N', m, n, l, alpha, t, h_copy)

    assert info == 0
    np.testing.assert_allclose(h_result, np.zeros((m, n)), rtol=1e-14)


def test_mb01zd_l0_triangular():
    """
    Test l=0: H is upper/lower triangular (no extra diagonals).

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    m, n, l = 4, 4, 0
    alpha = 1.0

    t = np.triu(np.random.randn(m, m)).astype(float, order='F')
    h = np.triu(np.random.randn(m, n)).astype(float, order='F')

    h_copy = h.copy(order='F')
    h_result, info = mb01zd('L', 'U', 'N', 'N', m, n, l, alpha, t, h_copy)

    assert info == 0
    expected = alpha * t @ h
    np.testing.assert_allclose(h_result, expected, rtol=1e-13)


def test_mb01zd_l2_hessenberg_like():
    """
    Test l=2: upper Hessenberg-like with 2 subdiagonals.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    m, n, l = 5, 4, 2
    alpha = 1.0

    t = np.triu(np.random.randn(m, m)).astype(float, order='F')

    h = np.zeros((m, n), order='F', dtype=float)
    for j in range(n):
        for i in range(min(j + l + 1, m)):
            h[i, j] = np.random.randn()

    h_copy = h.copy(order='F')
    h_result, info = mb01zd('L', 'U', 'N', 'N', m, n, l, alpha, t, h_copy)

    assert info == 0
    expected = alpha * t @ h
    np.testing.assert_allclose(h_result, expected, rtol=1e-13)


def test_mb01zd_empty_m0():
    """
    Test m=0 (empty case).
    """
    m, n, l = 0, 3, 0
    alpha = 1.0

    t = np.zeros((0, 0), order='F', dtype=float)
    h = np.zeros((0, 3), order='F', dtype=float)

    h_result, info = mb01zd('L', 'U', 'N', 'N', m, n, l, alpha, t, h)

    assert info == 0


def test_mb01zd_empty_n0():
    """
    Test n=0 (empty case).
    """
    m, n, l = 3, 0, 0
    alpha = 1.0

    t = np.eye(m, order='F', dtype=float)
    h = np.zeros((m, 0), order='F', dtype=float)

    h_result, info = mb01zd('L', 'U', 'N', 'N', m, n, l, alpha, t, h)

    assert info == 0


def test_mb01zd_invalid_side():
    """
    Test error handling for invalid SIDE parameter.
    """
    m, n, l = 3, 3, 1
    t = np.eye(m, order='F', dtype=float)
    h = np.random.randn(m, n).astype(float, order='F')

    h_result, info = mb01zd('X', 'U', 'N', 'N', m, n, l, 1.0, t, h)
    assert info == -1


def test_mb01zd_invalid_uplo():
    """
    Test error handling for invalid UPLO parameter.
    """
    m, n, l = 3, 3, 1
    t = np.eye(m, order='F', dtype=float)
    h = np.random.randn(m, n).astype(float, order='F')

    h_result, info = mb01zd('L', 'X', 'N', 'N', m, n, l, 1.0, t, h)
    assert info == -2


def test_mb01zd_invalid_trans():
    """
    Test error handling for invalid TRANS parameter.
    """
    m, n, l = 3, 3, 1
    t = np.eye(m, order='F', dtype=float)
    h = np.random.randn(m, n).astype(float, order='F')

    h_result, info = mb01zd('L', 'U', 'X', 'N', m, n, l, 1.0, t, h)
    assert info == -3


def test_mb01zd_invalid_diag():
    """
    Test error handling for invalid DIAG parameter.
    """
    m, n, l = 3, 3, 1
    t = np.eye(m, order='F', dtype=float)
    h = np.random.randn(m, n).astype(float, order='F')

    h_result, info = mb01zd('L', 'U', 'N', 'X', m, n, l, 1.0, t, h)
    assert info == -4


def test_mb01zd_invalid_l_upper():
    """
    Test error handling for invalid L parameter (UPLO='U').

    L must satisfy 0 <= L <= max(0, M-1).
    """
    m, n, l = 3, 3, 10
    t = np.eye(m, order='F', dtype=float)
    h = np.random.randn(m, n).astype(float, order='F')

    h_result, info = mb01zd('L', 'U', 'N', 'N', m, n, l, 1.0, t, h)
    assert info == -7


def test_mb01zd_invalid_l_lower():
    """
    Test error handling for invalid L parameter (UPLO='L').

    L must satisfy 0 <= L <= max(0, N-1).
    """
    m, n, l = 3, 3, 10
    t = np.tril(np.eye(m, order='F', dtype=float))
    h = np.random.randn(m, n).astype(float, order='F')

    h_result, info = mb01zd('L', 'L', 'N', 'N', m, n, l, 1.0, t, h)
    assert info == -7


def test_mb01zd_scaling_property():
    """
    Property test: (alpha*T)*H = alpha*(T*H).

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    m, n, l = 4, 3, 1
    alpha = 2.5

    t = np.triu(np.random.randn(m, m)).astype(float, order='F')

    h = np.zeros((m, n), order='F', dtype=float)
    for j in range(n):
        for i in range(min(j + l + 1, m)):
            h[i, j] = np.random.randn()

    h1 = h.copy(order='F')
    h_result1, info1 = mb01zd('L', 'U', 'N', 'N', m, n, l, alpha, t, h1)
    assert info1 == 0

    h2 = h.copy(order='F')
    h_result2, info2 = mb01zd('L', 'U', 'N', 'N', m, n, l, 1.0, t, h2)
    assert info2 == 0

    np.testing.assert_allclose(h_result1, alpha * h_result2, rtol=1e-14)


def test_mb01zd_identity_t():
    """
    Property test: I*H = H (identity matrix).

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    m, n, l = 4, 3, 1
    alpha = 1.0

    t = np.eye(m, order='F', dtype=float)

    h = np.zeros((m, n), order='F', dtype=float)
    for j in range(n):
        for i in range(min(j + l + 1, m)):
            h[i, j] = np.random.randn()

    h_copy = h.copy(order='F')
    h_result, info = mb01zd('L', 'U', 'N', 'N', m, n, l, alpha, t, h_copy)

    assert info == 0
    np.testing.assert_allclose(h_result, h, rtol=1e-14)


def test_mb01zd_associativity():
    """
    Property test: (T1*T2)*H = T1*(T2*H) - associativity.

    Random seed: 888 (for reproducibility)
    """
    np.random.seed(888)
    m, n, l = 4, 3, 1
    alpha = 1.0

    t1 = np.triu(np.random.randn(m, m)).astype(float, order='F')
    t2 = np.triu(np.random.randn(m, m)).astype(float, order='F')

    h = np.zeros((m, n), order='F', dtype=float)
    for j in range(n):
        for i in range(min(j + l + 1, m)):
            h[i, j] = np.random.randn()

    h1 = h.copy(order='F')
    h_temp, _ = mb01zd('L', 'U', 'N', 'N', m, n, l, alpha, t2, h1)
    h_temp = h_temp.copy(order='F')
    l_full = m - 1
    h_result1, _ = mb01zd('L', 'U', 'N', 'N', m, n, l_full, alpha, t1, h_temp)

    t_product = t1 @ t2
    for i in range(1, m):
        for j in range(i):
            t_product[i, j] = 0.0
    h2 = h.copy(order='F')
    expected = t_product @ h

    np.testing.assert_allclose(h_result1, expected, rtol=1e-12)


def test_mb01zd_square_m_eq_n():
    """
    Test square case m=n with various l values.

    Random seed: 999 (for reproducibility)
    """
    np.random.seed(999)
    m, n, l = 5, 5, 1
    alpha = 1.0

    t = np.triu(np.random.randn(m, m)).astype(float, order='F')

    h = np.zeros((m, n), order='F', dtype=float)
    for j in range(n):
        for i in range(min(j + l + 1, m)):
            h[i, j] = np.random.randn()

    h_copy = h.copy(order='F')
    h_result, info = mb01zd('L', 'U', 'N', 'N', m, n, l, alpha, t, h_copy)

    assert info == 0
    expected = alpha * t @ h
    np.testing.assert_allclose(h_result, expected, rtol=1e-13)


def test_mb01zd_rectangular_m_gt_n():
    """
    Test rectangular case m > n.

    Random seed: 1111 (for reproducibility)
    """
    np.random.seed(1111)
    m, n, l = 6, 3, 2
    alpha = 1.0

    t = np.triu(np.random.randn(m, m)).astype(float, order='F')

    h = np.zeros((m, n), order='F', dtype=float)
    for j in range(n):
        for i in range(min(j + l + 1, m)):
            h[i, j] = np.random.randn()

    h_copy = h.copy(order='F')
    h_result, info = mb01zd('L', 'U', 'N', 'N', m, n, l, alpha, t, h_copy)

    assert info == 0
    expected = alpha * t @ h
    np.testing.assert_allclose(h_result, expected, rtol=1e-13)


def test_mb01zd_rectangular_m_lt_n():
    """
    Test rectangular case m < n.

    Random seed: 2222 (for reproducibility)
    """
    np.random.seed(2222)
    m, n, l = 3, 6, 1
    alpha = 1.0

    t = np.triu(np.random.randn(m, m)).astype(float, order='F')

    h = np.zeros((m, n), order='F', dtype=float)
    for j in range(n):
        for i in range(min(j + l + 1, m)):
            h[i, j] = np.random.randn()

    h_copy = h.copy(order='F')
    h_result, info = mb01zd('L', 'U', 'N', 'N', m, n, l, alpha, t, h_copy)

    assert info == 0
    expected = alpha * t @ h
    np.testing.assert_allclose(h_result, expected, rtol=1e-13)
