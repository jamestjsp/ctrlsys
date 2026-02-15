"""
Tests for MB01RY: Compute triangle of R = alpha*R + beta*op(H)*B or beta*B*op(H).

MB01RY computes the upper or lower triangular part of:
    R := alpha*R + beta*op(H)*B  (SIDE='L')
    R := alpha*R + beta*B*op(H)  (SIDE='R')
where H is an upper Hessenberg matrix and op(H) = H or H'.
"""

import numpy as np
import pytest
from ctrlsys import mb01ry


def make_hessenberg(m):
    """Create a random upper Hessenberg matrix."""
    np.random.seed(444)
    h = np.random.randn(m, m).astype(float, order='F')
    for i in range(2, m):
        for j in range(i - 1):
            h[i, j] = 0.0
    return h


def test_mb01ry_left_notrans_upper():
    """
    Test SIDE='L', UPLO='U', TRANS='N': upper triangle of R = alpha*R + beta*H*B

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    m = 4
    alpha, beta = 1.5, 0.5

    r = np.random.randn(m, m).astype(float, order='F')
    r = (r + r.T) / 2

    h = make_hessenberg(m)
    b = np.random.randn(m, m).astype(float, order='F')

    r_orig = r.copy()
    r_expected = alpha * r_orig + beta * (h @ b)

    r_out, info = mb01ry('L', 'U', 'N', m, alpha, beta, r, h, b)

    assert info == 0
    np.testing.assert_allclose(np.triu(r_out), np.triu(r_expected), rtol=1e-14)


def test_mb01ry_left_trans_upper():
    """
    Test SIDE='L', UPLO='U', TRANS='T': upper triangle of R = alpha*R + beta*H'*B

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    m = 4
    alpha, beta = 2.0, 1.0

    r = np.random.randn(m, m).astype(float, order='F')
    r = (r + r.T) / 2

    h = make_hessenberg(m)
    b = np.random.randn(m, m).astype(float, order='F')

    r_orig = r.copy()
    r_expected = alpha * r_orig + beta * (h.T @ b)

    r_out, info = mb01ry('L', 'U', 'T', m, alpha, beta, r, h, b)

    assert info == 0
    np.testing.assert_allclose(np.triu(r_out), np.triu(r_expected), rtol=1e-14)


def test_mb01ry_right_notrans_upper():
    """
    Test SIDE='R', UPLO='U', TRANS='N': upper triangle of R = alpha*R + beta*B*H

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    m = 4
    alpha, beta = 1.0, 2.0

    r = np.random.randn(m, m).astype(float, order='F')
    r = (r + r.T) / 2

    h = make_hessenberg(m)
    b = np.random.randn(m, m).astype(float, order='F')

    r_orig = r.copy()
    r_expected = alpha * r_orig + beta * (b @ h)

    r_out, info = mb01ry('R', 'U', 'N', m, alpha, beta, r, h, b)

    assert info == 0
    np.testing.assert_allclose(np.triu(r_out), np.triu(r_expected), rtol=1e-13)


def test_mb01ry_right_trans_lower():
    """
    Test SIDE='R', UPLO='L', TRANS='T': lower triangle of R = alpha*R + beta*B*H'

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    m = 4
    alpha, beta = 0.5, 1.5

    r = np.random.randn(m, m).astype(float, order='F')
    r = (r + r.T) / 2

    h = make_hessenberg(m)
    b = np.random.randn(m, m).astype(float, order='F')

    r_orig = r.copy()
    r_expected = alpha * r_orig + beta * (b @ h.T)

    r_out, info = mb01ry('R', 'L', 'T', m, alpha, beta, r, h, b)

    assert info == 0
    np.testing.assert_allclose(np.tril(r_out), np.tril(r_expected), rtol=1e-14)


def test_mb01ry_left_notrans_lower():
    """
    Test SIDE='L', UPLO='L', TRANS='N': lower triangle of R = alpha*R + beta*H*B

    Random seed: 111 (for reproducibility)
    """
    np.random.seed(111)
    m = 4
    alpha, beta = 1.0, 1.0

    r = np.random.randn(m, m).astype(float, order='F')
    r = (r + r.T) / 2

    h = make_hessenberg(m)
    b = np.random.randn(m, m).astype(float, order='F')

    r_orig = r.copy()
    r_expected = alpha * r_orig + beta * (h @ b)

    r_out, info = mb01ry('L', 'L', 'N', m, alpha, beta, r, h, b)

    assert info == 0
    np.testing.assert_allclose(np.tril(r_out), np.tril(r_expected), rtol=1e-14)


def test_mb01ry_beta_zero():
    """
    Test beta=0: R = alpha*R (H and B ignored)

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    m = 3
    alpha, beta = 2.5, 0.0

    r = np.random.randn(m, m).astype(float, order='F')
    r = (r + r.T) / 2

    h = make_hessenberg(m)
    b = np.random.randn(m, m).astype(float, order='F')

    r_orig = r.copy()

    r_out, info = mb01ry('L', 'U', 'N', m, alpha, beta, r, h, b)

    assert info == 0
    np.testing.assert_allclose(np.triu(r_out), np.triu(alpha * r_orig), rtol=1e-14)


def test_mb01ry_alpha_zero():
    """
    Test alpha=0: R = beta*H*B (R input ignored)

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    m = 3
    alpha, beta = 0.0, 1.0

    r = np.ones((m, m), order='F', dtype=float) * 999.0

    h = make_hessenberg(m)
    b = np.random.randn(m, m).astype(float, order='F')

    r_expected = beta * (h @ b)

    r_out, info = mb01ry('L', 'U', 'N', m, alpha, beta, r, h, b)

    assert info == 0
    np.testing.assert_allclose(np.triu(r_out), np.triu(r_expected), rtol=1e-14)


def test_mb01ry_alpha_beta_zero():
    """
    Test alpha=0, beta=0: R = 0
    """
    m = 3
    alpha, beta = 0.0, 0.0

    r = np.ones((m, m), order='F', dtype=float) * 999.0
    h = np.eye(m, order='F', dtype=float)
    b = np.eye(m, order='F', dtype=float)

    r_out, info = mb01ry('L', 'U', 'N', m, alpha, beta, r, h, b)

    assert info == 0
    np.testing.assert_allclose(np.triu(r_out), np.zeros((m, m)), rtol=1e-15)


def test_mb01ry_m_zero():
    """
    Test quick return for m=0
    """
    m = 0
    alpha, beta = 1.0, 1.0

    r = np.array([], order='F', dtype=float).reshape(0, 0)
    h = np.array([], order='F', dtype=float).reshape(0, 0)
    b = np.array([], order='F', dtype=float).reshape(0, 0)

    r_out, info = mb01ry('L', 'U', 'N', m, alpha, beta, r, h, b)

    assert info == 0


def test_mb01ry_small_2x2():
    """
    Test small 2x2 case.

    Random seed: 555 (for reproducibility)
    """
    np.random.seed(555)
    m = 2
    alpha, beta = 1.0, 1.0

    r = np.array([
        [1.0, 2.0],
        [2.0, 3.0]
    ], order='F', dtype=float)

    h = np.array([
        [1.0, 2.0],
        [0.5, 3.0]
    ], order='F', dtype=float)

    b = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], order='F', dtype=float)

    r_orig = r.copy()
    r_expected = alpha * r_orig + beta * (h @ b)

    r_out, info = mb01ry('L', 'U', 'N', m, alpha, beta, r, h, b)

    assert info == 0
    np.testing.assert_allclose(np.triu(r_out), np.triu(r_expected), rtol=1e-14)


def test_mb01ry_h_restored():
    """
    Validate that H is restored after the call (subdiagonal elements).

    Random seed: 666 (for reproducibility)
    """
    np.random.seed(666)
    m = 4
    alpha, beta = 1.0, 1.0

    r = np.random.randn(m, m).astype(float, order='F')

    h = make_hessenberg(m)
    h_orig = h.copy()

    b = np.random.randn(m, m).astype(float, order='F')

    r_out, info = mb01ry('L', 'U', 'N', m, alpha, beta, r, h, b)

    assert info == 0
    np.testing.assert_allclose(h, h_orig, rtol=1e-15)


def test_mb01ry_invalid_side():
    """
    Test error handling for invalid SIDE parameter
    """
    m = 2
    r = np.eye(m, order='F', dtype=float)
    h = np.eye(m, order='F', dtype=float)
    b = np.eye(m, order='F', dtype=float)

    r_out, info = mb01ry('X', 'U', 'N', m, 1.0, 1.0, r, h, b)

    assert info == -1


def test_mb01ry_invalid_uplo():
    """
    Test error handling for invalid UPLO parameter
    """
    m = 2
    r = np.eye(m, order='F', dtype=float)
    h = np.eye(m, order='F', dtype=float)
    b = np.eye(m, order='F', dtype=float)

    r_out, info = mb01ry('L', 'X', 'N', m, 1.0, 1.0, r, h, b)

    assert info == -2
