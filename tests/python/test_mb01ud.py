"""
Tests for MB01UD: Compute B = alpha*op(H)*A or B = alpha*A*op(H).

MB01UD computes:
    B = alpha*op(H)*A  (SIDE='L')
    B = alpha*A*op(H)  (SIDE='R')
where H is an upper Hessenberg matrix and op(H) = H or H'.
"""

import numpy as np
import pytest
from ctrlsys import mb01ud


def make_hessenberg(k):
    """Create a random upper Hessenberg matrix."""
    h = np.random.randn(k, k).astype(float, order='F')
    for i in range(2, k):
        for j in range(i - 1):
            h[i, j] = 0.0
    return h


def test_mb01ud_left_notrans():
    """
    Test SIDE='L', TRANS='N': B = alpha*H*A

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    m, n = 4, 3
    alpha = 2.0

    h = make_hessenberg(m)
    a = np.random.randn(m, n).astype(float, order='F')

    b_expected = alpha * (h @ a)

    b_out, info = mb01ud('L', 'N', m, n, alpha, h, a)

    assert info == 0
    np.testing.assert_allclose(b_out, b_expected, rtol=1e-14)


def test_mb01ud_left_trans():
    """
    Test SIDE='L', TRANS='T': B = alpha*H'*A

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    m, n = 4, 3
    alpha = 1.5

    h = make_hessenberg(m)
    a = np.random.randn(m, n).astype(float, order='F')

    b_expected = alpha * (h.T @ a)

    b_out, info = mb01ud('L', 'T', m, n, alpha, h, a)

    assert info == 0
    np.testing.assert_allclose(b_out, b_expected, rtol=1e-14)


def test_mb01ud_right_notrans():
    """
    Test SIDE='R', TRANS='N': B = alpha*A*H

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    m, n = 3, 4
    alpha = 0.5

    h = make_hessenberg(n)
    a = np.random.randn(m, n).astype(float, order='F')

    b_expected = alpha * (a @ h)

    b_out, info = mb01ud('R', 'N', m, n, alpha, h, a)

    assert info == 0
    np.testing.assert_allclose(b_out, b_expected, rtol=1e-14)


def test_mb01ud_right_trans():
    """
    Test SIDE='R', TRANS='T': B = alpha*A*H'

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    m, n = 3, 4
    alpha = 3.0

    h = make_hessenberg(n)
    a = np.random.randn(m, n).astype(float, order='F')

    b_expected = alpha * (a @ h.T)

    b_out, info = mb01ud('R', 'T', m, n, alpha, h, a)

    assert info == 0
    np.testing.assert_allclose(b_out, b_expected, rtol=1e-14)


def test_mb01ud_alpha_zero():
    """
    Test alpha=0: B = 0 (H and A ignored)
    """
    np.random.seed(111)
    m, n = 3, 4
    alpha = 0.0

    h = make_hessenberg(m)
    a = np.random.randn(m, n).astype(float, order='F')

    b_out, info = mb01ud('L', 'N', m, n, alpha, h, a)

    assert info == 0
    np.testing.assert_allclose(b_out, np.zeros((m, n)), rtol=1e-15)


def test_mb01ud_m_zero():
    """
    Test quick return for m=0
    """
    m, n = 0, 3
    alpha = 1.0

    h = np.array([], order='F', dtype=float).reshape(0, 0)
    a = np.array([], order='F', dtype=float).reshape(0, n)

    b_out, info = mb01ud('L', 'N', m, n, alpha, h, a)

    assert info == 0


def test_mb01ud_n_zero():
    """
    Test quick return for n=0
    """
    m, n = 3, 0
    alpha = 1.0

    h = make_hessenberg(m)
    a = np.array([], order='F', dtype=float).reshape(m, 0)

    b_out, info = mb01ud('L', 'N', m, n, alpha, h, a)

    assert info == 0


def test_mb01ud_alpha_one():
    """
    Test alpha=1: B = H*A

    Random seed: 222 (for reproducibility)
    """
    np.random.seed(222)
    m, n = 4, 4
    alpha = 1.0

    h = make_hessenberg(m)
    a = np.random.randn(m, n).astype(float, order='F')

    b_expected = h @ a

    b_out, info = mb01ud('L', 'N', m, n, alpha, h, a)

    assert info == 0
    np.testing.assert_allclose(b_out, b_expected, rtol=1e-13)


def test_mb01ud_h_restored():
    """
    Validate that H is restored after the call.

    Random seed: 333 (for reproducibility)
    """
    np.random.seed(333)
    m, n = 4, 3
    alpha = 1.0

    h = make_hessenberg(m)
    h_orig = h.copy()
    a = np.random.randn(m, n).astype(float, order='F')

    b_out, info = mb01ud('L', 'N', m, n, alpha, h, a)

    assert info == 0
    np.testing.assert_allclose(h, h_orig, rtol=1e-15)


def test_mb01ud_invalid_side():
    """
    Test error handling for invalid SIDE parameter
    """
    m, n = 2, 2
    h = np.eye(m, order='F', dtype=float)
    a = np.eye(m, n, order='F', dtype=float)

    b_out, info = mb01ud('X', 'N', m, n, 1.0, h, a)

    assert info == -1


def test_mb01ud_invalid_trans():
    """
    Test error handling for invalid TRANS parameter
    """
    m, n = 2, 2
    h = np.eye(m, order='F', dtype=float)
    a = np.eye(m, n, order='F', dtype=float)

    b_out, info = mb01ud('L', 'X', m, n, 1.0, h, a)

    assert info == -2
