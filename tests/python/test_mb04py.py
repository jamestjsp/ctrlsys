"""
Tests for MB04PY - Apply elementary reflector to matrix from left or right

MB04PY applies real elementary reflector H to real m-by-n matrix C:
    H * C  (side='L')
    C * H  (side='R')

H = I - tau * u * u',  u = [1; v]

where tau is scalar, v is (m-1)-vector for left, (n-1)-vector for right.

If tau=0, H is identity matrix.
In-line code used if H has order < 11.
"""

import numpy as np
import pytest


def apply_householder_left(c, v, tau):
    """
    Reference: H * C where H = I - tau*u*u', u = [1; v]

    Args:
        c: m-by-n matrix (modified copy)
        v: (m-1)-vector
        tau: scalar

    Returns:
        H * C
    """
    m, n = c.shape
    if tau == 0.0:
        return c.copy()

    u = np.vstack([[1.0], v.reshape(-1, 1)])
    h = np.eye(m) - tau * (u @ u.T)
    return h @ c


def apply_householder_right(c, v, tau):
    """
    Reference: C * H where H = I - tau*u*u', u = [1; v]

    Args:
        c: m-by-n matrix (modified copy)
        v: (n-1)-vector
        tau: scalar

    Returns:
        C * H
    """
    m, n = c.shape
    if tau == 0.0:
        return c.copy()

    u = np.vstack([[1.0], v.reshape(-1, 1)])
    h = np.eye(n) - tau * (u @ u.T)
    return c @ h


"""Tests for SIDE='L' (H * C)"""

def test_left_1x1():
    """
    1x1 Householder from left (m=1)

    For m=1, v is empty, u=[1], H = I - tau
    So C := (1-tau)*C
    Random seed: 42
    """
    np.random.seed(42)

    m, n = 1, 3
    c = np.random.randn(m, n).astype(float, order='F')
    v = np.empty(0, dtype=float, order='F')
    tau = 1.5

    c_expected = (1.0 - tau) * c

    from ctrlsys import mb04py

    c_out = c.copy(order='F')
    mb04py('L', m, n, v, tau, c_out)

    np.testing.assert_allclose(c_out, c_expected, rtol=1e-14)

def test_left_2x2():
    """
    2x2 Householder from left (m=2)

    Tests inline code for order 2.
    Random seed: 43
    """
    np.random.seed(43)

    m, n = 2, 4
    c = np.random.randn(m, n).astype(float, order='F')
    v = np.random.randn(m - 1).astype(float, order='F')
    tau = 1.8

    c_expected = apply_householder_left(c, v, tau)

    from ctrlsys import mb04py

    c_out = c.copy(order='F')
    mb04py('L', m, n, v, tau, c_out)

    np.testing.assert_allclose(c_out, c_expected, rtol=1e-14)

def test_left_3x3():
    """
    3x3 Householder from left (m=3)

    Tests inline code for order 3.
    Random seed: 44
    """
    np.random.seed(44)

    m, n = 3, 5
    c = np.random.randn(m, n).astype(float, order='F')
    v = np.random.randn(m - 1).astype(float, order='F')
    tau = 1.6

    c_expected = apply_householder_left(c, v, tau)

    from ctrlsys import mb04py

    c_out = c.copy(order='F')
    mb04py('L', m, n, v, tau, c_out)

    np.testing.assert_allclose(c_out, c_expected, rtol=1e-14)

def test_left_5x5():
    """
    5x5 Householder from left (m=5)

    Random seed: 45
    """
    np.random.seed(45)

    m, n = 5, 6
    c = np.random.randn(m, n).astype(float, order='F')
    v = np.random.randn(m - 1).astype(float, order='F')
    tau = 1.7

    c_expected = apply_householder_left(c, v, tau)

    from ctrlsys import mb04py

    c_out = c.copy(order='F')
    mb04py('L', m, n, v, tau, c_out)

    np.testing.assert_allclose(c_out, c_expected, rtol=1e-14)

def test_left_10x10():
    """
    10x10 Householder from left (m=10)

    Maximum inline order.
    Random seed: 46
    """
    np.random.seed(46)

    m, n = 10, 8
    c = np.random.randn(m, n).astype(float, order='F')
    v = np.random.randn(m - 1).astype(float, order='F')
    tau = 1.9

    c_expected = apply_householder_left(c, v, tau)

    from ctrlsys import mb04py

    c_out = c.copy(order='F')
    mb04py('L', m, n, v, tau, c_out)

    np.testing.assert_allclose(c_out, c_expected, rtol=1e-14)

def test_left_general_blas():
    """
    Large Householder from left (m=15) using BLAS

    Tests general code path for order >= 11.
    Random seed: 47
    """
    np.random.seed(47)

    m, n = 15, 10
    c = np.random.randn(m, n).astype(float, order='F')
    v = np.random.randn(m - 1).astype(float, order='F')
    tau = 1.5

    c_expected = apply_householder_left(c, v, tau)

    from ctrlsys import mb04py

    c_out = c.copy(order='F')
    mb04py('L', m, n, v, tau, c_out)

    np.testing.assert_allclose(c_out, c_expected, rtol=1e-13)


"""Tests for SIDE='R' (C * H)"""

def test_right_1x1():
    """
    1x1 Householder from right (n=1)

    For n=1, v is empty, u=[1], H = I - tau
    So C := C*(1-tau)
    Random seed: 52
    """
    np.random.seed(52)

    m, n = 4, 1
    c = np.random.randn(m, n).astype(float, order='F')
    v = np.empty(0, dtype=float, order='F')
    tau = 1.5

    c_expected = c * (1.0 - tau)

    from ctrlsys import mb04py

    c_out = c.copy(order='F')
    mb04py('R', m, n, v, tau, c_out)

    np.testing.assert_allclose(c_out, c_expected, rtol=1e-14)

def test_right_2x2():
    """
    2x2 Householder from right (n=2)

    Tests inline code for order 2.
    Random seed: 53
    """
    np.random.seed(53)

    m, n = 5, 2
    c = np.random.randn(m, n).astype(float, order='F')
    v = np.random.randn(n - 1).astype(float, order='F')
    tau = 1.8

    c_expected = apply_householder_right(c, v, tau)

    from ctrlsys import mb04py

    c_out = c.copy(order='F')
    mb04py('R', m, n, v, tau, c_out)

    np.testing.assert_allclose(c_out, c_expected, rtol=1e-14)

def test_right_3x3():
    """
    3x3 Householder from right (n=3)

    Tests inline code for order 3.
    Random seed: 54
    """
    np.random.seed(54)

    m, n = 6, 3
    c = np.random.randn(m, n).astype(float, order='F')
    v = np.random.randn(n - 1).astype(float, order='F')
    tau = 1.6

    c_expected = apply_householder_right(c, v, tau)

    from ctrlsys import mb04py

    c_out = c.copy(order='F')
    mb04py('R', m, n, v, tau, c_out)

    np.testing.assert_allclose(c_out, c_expected, rtol=1e-14)

def test_right_5x5():
    """
    5x5 Householder from right (n=5)

    Random seed: 55
    """
    np.random.seed(55)

    m, n = 7, 5
    c = np.random.randn(m, n).astype(float, order='F')
    v = np.random.randn(n - 1).astype(float, order='F')
    tau = 1.7

    c_expected = apply_householder_right(c, v, tau)

    from ctrlsys import mb04py

    c_out = c.copy(order='F')
    mb04py('R', m, n, v, tau, c_out)

    np.testing.assert_allclose(c_out, c_expected, rtol=1e-14)

def test_right_10x10():
    """
    10x10 Householder from right (n=10)

    Maximum inline order.
    Random seed: 56
    """
    np.random.seed(56)

    m, n = 8, 10
    c = np.random.randn(m, n).astype(float, order='F')
    v = np.random.randn(n - 1).astype(float, order='F')
    tau = 1.9

    c_expected = apply_householder_right(c, v, tau)

    from ctrlsys import mb04py

    c_out = c.copy(order='F')
    mb04py('R', m, n, v, tau, c_out)

    np.testing.assert_allclose(c_out, c_expected, rtol=1e-14)

def test_right_general_blas():
    """
    Large Householder from right (n=15) using BLAS

    Tests general code path for order >= 11.
    Random seed: 57
    """
    np.random.seed(57)

    m, n = 10, 15
    c = np.random.randn(m, n).astype(float, order='F')
    v = np.random.randn(n - 1).astype(float, order='F')
    tau = 1.5

    c_expected = apply_householder_right(c, v, tau)

    from ctrlsys import mb04py

    c_out = c.copy(order='F')
    mb04py('R', m, n, v, tau, c_out)

    np.testing.assert_allclose(c_out, c_expected, rtol=1e-13)


"""Edge cases and special values"""

def test_tau_zero_left():
    """
    tau=0: H=I, matrix unchanged (left application)

    Random seed: 60
    """
    np.random.seed(60)

    m, n = 5, 4
    c = np.random.randn(m, n).astype(float, order='F')
    v = np.random.randn(m - 1).astype(float, order='F')
    tau = 0.0

    c_expected = c.copy()

    from ctrlsys import mb04py

    c_out = c.copy(order='F')
    mb04py('L', m, n, v, tau, c_out)

    np.testing.assert_allclose(c_out, c_expected, rtol=1e-14)

def test_tau_zero_right():
    """
    tau=0: H=I, matrix unchanged (right application)

    Random seed: 61
    """
    np.random.seed(61)

    m, n = 4, 5
    c = np.random.randn(m, n).astype(float, order='F')
    v = np.random.randn(n - 1).astype(float, order='F')
    tau = 0.0

    c_expected = c.copy()

    from ctrlsys import mb04py

    c_out = c.copy(order='F')
    mb04py('R', m, n, v, tau, c_out)

    np.testing.assert_allclose(c_out, c_expected, rtol=1e-14)

def test_single_column_left():
    """
    n=1: Single column, left application

    Random seed: 62
    """
    np.random.seed(62)

    m, n = 5, 1
    c = np.random.randn(m, n).astype(float, order='F')
    v = np.random.randn(m - 1).astype(float, order='F')
    tau = 1.5

    c_expected = apply_householder_left(c, v, tau)

    from ctrlsys import mb04py

    c_out = c.copy(order='F')
    mb04py('L', m, n, v, tau, c_out)

    np.testing.assert_allclose(c_out, c_expected, rtol=1e-14)

def test_single_row_right():
    """
    m=1: Single row, right application

    Random seed: 63
    """
    np.random.seed(63)

    m, n = 1, 5
    c = np.random.randn(m, n).astype(float, order='F')
    v = np.random.randn(n - 1).astype(float, order='F')
    tau = 1.5

    c_expected = apply_householder_right(c, v, tau)

    from ctrlsys import mb04py

    c_out = c.copy(order='F')
    mb04py('R', m, n, v, tau, c_out)

    np.testing.assert_allclose(c_out, c_expected, rtol=1e-14)


"""Mathematical property tests"""

def test_orthogonality_left():
    """
    H is orthogonal: H'*H = I

    Verify by applying H twice: H*(H*C) = C
    Random seed: 70
    """
    np.random.seed(70)

    m, n = 5, 4
    c = np.random.randn(m, n).astype(float, order='F')
    v = np.random.randn(m - 1).astype(float, order='F')
    tau = 2.0 / (1.0 + np.dot(v, v))

    from ctrlsys import mb04py

    c_out = c.copy(order='F')
    mb04py('L', m, n, v, tau, c_out)
    mb04py('L', m, n, v, tau, c_out)

    np.testing.assert_allclose(c_out, c, rtol=1e-14, atol=1e-15)

def test_orthogonality_right():
    """
    H is orthogonal: H*H' = I

    Verify by applying H twice: (C*H)*H = C
    Random seed: 71
    """
    np.random.seed(71)

    m, n = 4, 5
    c = np.random.randn(m, n).astype(float, order='F')
    v = np.random.randn(n - 1).astype(float, order='F')
    tau = 2.0 / (1.0 + np.dot(v, v))

    from ctrlsys import mb04py

    c_out = c.copy(order='F')
    mb04py('R', m, n, v, tau, c_out)
    mb04py('R', m, n, v, tau, c_out)

    np.testing.assert_allclose(c_out, c, rtol=1e-14, atol=1e-15)

def test_reflection_property():
    """
    H*u = -u when tau = 2/(u'*u)

    Verify reflector flips sign of u.
    Random seed: 72
    """
    np.random.seed(72)

    m = 5
    v = np.random.randn(m - 1).astype(float, order='F')
    u = np.vstack([[1.0], v.reshape(-1, 1)])
    tau = 2.0 / (1.0 + np.dot(v, v))

    c = u.copy().astype(float, order='F')

    from ctrlsys import mb04py

    mb04py('L', m, 1, v, tau, c)

    np.testing.assert_allclose(c, -u, rtol=1e-14, atol=1e-15)

def test_frobenius_norm_preserved():
    """
    Orthogonal transformations preserve Frobenius norm.

    ||H*C||_F = ||C||_F
    Random seed: 73
    """
    np.random.seed(73)

    m, n = 6, 5
    c = np.random.randn(m, n).astype(float, order='F')
    v = np.random.randn(m - 1).astype(float, order='F')
    tau = 2.0 / (1.0 + np.dot(v, v))

    norm_before = np.linalg.norm(c, 'fro')

    from ctrlsys import mb04py

    c_out = c.copy(order='F')
    mb04py('L', m, n, v, tau, c_out)

    norm_after = np.linalg.norm(c_out, 'fro')

    np.testing.assert_allclose(norm_after, norm_before, rtol=1e-14)
