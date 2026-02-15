"""
Tests for MB04OY - Apply Householder reflector to [A; B]

MB04OY applies elementary reflector H to (m+1)-by-n matrix C = [A; B],
where A has one row, from the left:

    H = I - tau * u * u',  u = [1; v]

where tau is scalar and v is m-vector.

Special inline code for order < 11, general BLAS code for larger orders.
"""

import numpy as np
import pytest


def apply_householder_reference(a, b, v, tau):
    """
    Reference implementation: H * [A; B] where H = I - tau*u*u', u = [1; v]

    Args:
        a: 1-by-n array (first row)
        b: m-by-n array (remaining rows)
        v: m-vector
        tau: scalar

    Returns:
        a_out, b_out: Updated matrices
    """
    m, n = b.shape

    if tau == 0.0:
        return a.copy(), b.copy()

    c = np.vstack([a, b])
    u = np.vstack([[1.0], v.reshape(-1, 1)])

    h = np.eye(m + 1) - tau * (u @ u.T)
    c_out = h @ c

    return c_out[0:1, :], c_out[1:, :]


def test_mb04oy_basic_2x2():
    """
    Basic test: 2x2 Householder (m=1, n=3)

    Tests inline code path for small reflector.
    Random seed: 42
    """
    np.random.seed(42)

    m, n = 1, 3
    a = np.random.randn(1, n).astype(float, order='F')
    b = np.random.randn(m, n).astype(float, order='F')
    v = np.random.randn(m).astype(float, order='F')
    tau = 1.5

    a_expected, b_expected = apply_householder_reference(a, b, v, tau)

    from ctrlsys import mb04oy

    a_out = a.copy(order='F')
    b_out = b.copy(order='F')
    mb04oy(m, n, v, tau, a_out, b_out)

    np.testing.assert_allclose(a_out, a_expected, rtol=1e-14)
    np.testing.assert_allclose(b_out, b_expected, rtol=1e-14)


def test_mb04oy_3x3():
    """
    Test 3x3 Householder (m=2, n=4)

    Tests inline code for order 3.
    Random seed: 123
    """
    np.random.seed(123)

    m, n = 2, 4
    a = np.random.randn(1, n).astype(float, order='F')
    b = np.random.randn(m, n).astype(float, order='F')
    v = np.random.randn(m).astype(float, order='F')
    tau = 2.0

    a_expected, b_expected = apply_householder_reference(a, b, v, tau)

    from ctrlsys import mb04oy

    a_out = a.copy(order='F')
    b_out = b.copy(order='F')
    mb04oy(m, n, v, tau, a_out, b_out)

    np.testing.assert_allclose(a_out, a_expected, rtol=1e-14)
    np.testing.assert_allclose(b_out, b_expected, rtol=1e-14)


def test_mb04oy_5x5():
    """
    Test 5x5 Householder (m=4, n=6)

    Tests inline code for order 5.
    Random seed: 456
    """
    np.random.seed(456)

    m, n = 4, 6
    a = np.random.randn(1, n).astype(float, order='F')
    b = np.random.randn(m, n).astype(float, order='F')
    v = np.random.randn(m).astype(float, order='F')
    tau = 1.8

    a_expected, b_expected = apply_householder_reference(a, b, v, tau)

    from ctrlsys import mb04oy

    a_out = a.copy(order='F')
    b_out = b.copy(order='F')
    mb04oy(m, n, v, tau, a_out, b_out)

    np.testing.assert_allclose(a_out, a_expected, rtol=1e-13)
    np.testing.assert_allclose(b_out, b_expected, rtol=1e-13)


def test_mb04oy_10x10():
    """
    Test 10x10 Householder (m=9, n=5)

    Tests inline code for maximum inline order.
    Random seed: 789
    """
    np.random.seed(789)

    m, n = 9, 5
    a = np.random.randn(1, n).astype(float, order='F')
    b = np.random.randn(m, n).astype(float, order='F')
    v = np.random.randn(m).astype(float, order='F')
    tau = 1.6

    a_expected, b_expected = apply_householder_reference(a, b, v, tau)

    from ctrlsys import mb04oy

    a_out = a.copy(order='F')
    b_out = b.copy(order='F')
    mb04oy(m, n, v, tau, a_out, b_out)

    np.testing.assert_allclose(a_out, a_expected, rtol=1e-14)
    np.testing.assert_allclose(b_out, b_expected, rtol=1e-14)


def test_mb04oy_large_general():
    """
    Test large Householder (m=15, n=8) using BLAS code

    Tests general BLAS path for order >= 11.
    Random seed: 888
    """
    np.random.seed(888)

    m, n = 15, 8
    a = np.random.randn(1, n).astype(float, order='F')
    b = np.random.randn(m, n).astype(float, order='F')
    v = np.random.randn(m).astype(float, order='F')
    tau = 1.9

    a_expected, b_expected = apply_householder_reference(a, b, v, tau)

    from ctrlsys import mb04oy

    a_out = a.copy(order='F')
    b_out = b.copy(order='F')
    mb04oy(m, n, v, tau, a_out, b_out)

    np.testing.assert_allclose(a_out, a_expected, rtol=1e-13)
    np.testing.assert_allclose(b_out, b_expected, rtol=1e-13)


def test_mb04oy_tau_zero():
    """
    Test tau=0 case (identity transformation)

    When tau=0, H=I, so matrices should be unchanged.
    Random seed: 999
    """
    np.random.seed(999)

    m, n = 5, 4
    a = np.random.randn(1, n).astype(float, order='F')
    b = np.random.randn(m, n).astype(float, order='F')
    v = np.random.randn(m).astype(float, order='F')
    tau = 0.0

    a_expected = a.copy()
    b_expected = b.copy()

    from ctrlsys import mb04oy

    a_out = a.copy(order='F')
    b_out = b.copy(order='F')
    mb04oy(m, n, v, tau, a_out, b_out)

    np.testing.assert_allclose(a_out, a_expected, rtol=1e-14)
    np.testing.assert_allclose(b_out, b_expected, rtol=1e-14)


def test_mb04oy_orthogonality():
    """
    Mathematical property: H is orthogonal (H'*H = I)

    Verify orthogonality of Householder reflector.
    Random seed: 111
    """
    np.random.seed(111)

    m, n = 6, 5
    v = np.random.randn(m).astype(float, order='F')
    tau = 2.0 / (1.0 + np.dot(v, v))

    u = np.vstack([[1.0], v.reshape(-1, 1)])
    h = np.eye(m + 1) - tau * (u @ u.T)

    identity = h @ h.T
    np.testing.assert_allclose(identity, np.eye(m + 1), rtol=1e-14, atol=1e-15)


def test_mb04oy_m_zero():
    """
    Edge case: m=0 (only A row, B is empty)

    Tests 1x1 Householder: A := (1-tau)*A
    Random seed: 222
    """
    np.random.seed(222)

    m, n = 0, 4
    a = np.random.randn(1, n).astype(float, order='F')
    b = np.empty((0, n), dtype=float, order='F')
    v = np.empty(0, dtype=float, order='F')
    tau = 1.5

    a_expected = (1.0 - tau) * a

    from ctrlsys import mb04oy

    a_out = a.copy(order='F')
    b_out = b.copy(order='F')
    mb04oy(m, n, v, tau, a_out, b_out)

    np.testing.assert_allclose(a_out, a_expected, rtol=1e-14)


def test_mb04oy_n_zero():
    """
    Edge case: n=0 (no columns)

    Should handle gracefully (no operations).
    """
    m, n = 3, 0
    a = np.empty((1, 0), dtype=float, order='F')
    b = np.empty((m, 0), dtype=float, order='F')
    v = np.array([1.0, 2.0, 3.0], dtype=float, order='F')
    tau = 1.5

    from ctrlsys import mb04oy

    a_out = a.copy(order='F')
    b_out = b.copy(order='F')
    mb04oy(m, n, v, tau, a_out, b_out)

    assert a_out.shape == (1, 0)
    assert b_out.shape == (m, 0)


def test_mb04oy_single_column():
    """
    Edge case: n=1 (single column vector)

    Random seed: 333
    """
    np.random.seed(333)

    m, n = 7, 1
    a = np.random.randn(1, n).astype(float, order='F')
    b = np.random.randn(m, n).astype(float, order='F')
    v = np.random.randn(m).astype(float, order='F')
    tau = 1.4

    a_expected, b_expected = apply_householder_reference(a, b, v, tau)

    from ctrlsys import mb04oy

    a_out = a.copy(order='F')
    b_out = b.copy(order='F')
    mb04oy(m, n, v, tau, a_out, b_out)

    np.testing.assert_allclose(a_out, a_expected, rtol=1e-14)
    np.testing.assert_allclose(b_out, b_expected, rtol=1e-14)
