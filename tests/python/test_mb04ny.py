"""
Tests for MB04NY - Apply Householder reflector to [A B] from the right.

MB04NY applies elementary reflector H to real m-by-(n+1) matrix C = [A B],
where A has one column, from the right:

    H = I - tau * u * u',  u = [1; v]

where tau is scalar and v is n-vector.

The transformation: C := C * H

Special inline code for order < 11, general BLAS code for larger orders.
"""

import numpy as np
import pytest


def apply_householder_right_reference(a, b, v, incv, tau):
    """
    Reference implementation: [A B] * H where H = I - tau*u*u', u = [1; v]

    Args:
        a: m-by-1 array (first column)
        b: m-by-n array (remaining columns)
        v: n-vector with increment incv
        incv: increment for v elements
        tau: scalar

    Returns:
        a_out, b_out: Updated matrices
    """
    m = a.shape[0]
    n = b.shape[1]

    if tau == 0.0:
        return a.copy(), b.copy()

    # Extract v elements with increment
    if incv > 0:
        v_full = v[::incv][:n]
    else:
        # Negative increment: start from end
        start = (1 - n) * incv
        v_full = v[start::incv][:n]

    # C = [A B] is m x (n+1), u = [1; v] is (n+1) x 1
    # C * u = A*1 + B*v = A + B @ v
    c_times_u = a.flatten() + b @ v_full  # m-vector

    # C - tau * (C*u) * u' = [A B] - tau * (c_times_u) @ [1, v']
    a_out = a - tau * c_times_u.reshape(-1, 1)
    b_out = b - tau * np.outer(c_times_u, v_full)

    return a_out, b_out


def test_mb04ny_basic_2x2():
    """
    Basic test: 2x2 Householder (n=1, m=3)

    Tests inline code path for small reflector.
    Random seed: 42
    """
    np.random.seed(42)

    m, n = 3, 1
    a = np.random.randn(m, 1).astype(float, order='F')
    b = np.random.randn(m, n).astype(float, order='F')
    v = np.random.randn(n).astype(float, order='F')
    incv = 1
    tau = 1.5

    a_expected, b_expected = apply_householder_right_reference(a, b, v, incv, tau)

    from ctrlsys import _slicot

    a_out = a.copy(order='F')
    b_out = b.copy(order='F')
    _ctrlsys.mb04ny(m, n, v, incv, tau, a_out, b_out)

    np.testing.assert_allclose(a_out, a_expected, rtol=1e-14)
    np.testing.assert_allclose(b_out, b_expected, rtol=1e-14)


def test_mb04ny_3x3():
    """
    Test 3x3 Householder (n=2, m=4)

    Tests inline code for order 3.
    Random seed: 123
    """
    np.random.seed(123)

    m, n = 4, 2
    a = np.random.randn(m, 1).astype(float, order='F')
    b = np.random.randn(m, n).astype(float, order='F')
    v = np.random.randn(n).astype(float, order='F')
    incv = 1
    tau = 2.0

    a_expected, b_expected = apply_householder_right_reference(a, b, v, incv, tau)

    from ctrlsys import _slicot

    a_out = a.copy(order='F')
    b_out = b.copy(order='F')
    _ctrlsys.mb04ny(m, n, v, incv, tau, a_out, b_out)

    np.testing.assert_allclose(a_out, a_expected, rtol=1e-14)
    np.testing.assert_allclose(b_out, b_expected, rtol=1e-14)


def test_mb04ny_5x5():
    """
    Test 5x5 Householder (n=4, m=6)

    Tests inline code for order 5.
    Random seed: 456
    """
    np.random.seed(456)

    m, n = 6, 4
    a = np.random.randn(m, 1).astype(float, order='F')
    b = np.random.randn(m, n).astype(float, order='F')
    v = np.random.randn(n).astype(float, order='F')
    incv = 1
    tau = 1.8

    a_expected, b_expected = apply_householder_right_reference(a, b, v, incv, tau)

    from ctrlsys import _slicot

    a_out = a.copy(order='F')
    b_out = b.copy(order='F')
    _ctrlsys.mb04ny(m, n, v, incv, tau, a_out, b_out)

    np.testing.assert_allclose(a_out, a_expected, rtol=1e-13)
    np.testing.assert_allclose(b_out, b_expected, rtol=1e-13)


def test_mb04ny_10x10():
    """
    Test 10x10 Householder (n=9, m=5)

    Tests inline code for maximum inline order.
    Random seed: 789
    """
    np.random.seed(789)

    m, n = 5, 9
    a = np.random.randn(m, 1).astype(float, order='F')
    b = np.random.randn(m, n).astype(float, order='F')
    v = np.random.randn(n).astype(float, order='F')
    incv = 1
    tau = 1.6

    a_expected, b_expected = apply_householder_right_reference(a, b, v, incv, tau)

    from ctrlsys import _slicot

    a_out = a.copy(order='F')
    b_out = b.copy(order='F')
    _ctrlsys.mb04ny(m, n, v, incv, tau, a_out, b_out)

    np.testing.assert_allclose(a_out, a_expected, rtol=1e-12)
    np.testing.assert_allclose(b_out, b_expected, rtol=1e-12)


def test_mb04ny_large_general():
    """
    Test large Householder (n=15, m=8) using BLAS code

    Tests general BLAS path for order >= 11.
    Random seed: 888
    """
    np.random.seed(888)

    m, n = 8, 15
    a = np.random.randn(m, 1).astype(float, order='F')
    b = np.random.randn(m, n).astype(float, order='F')
    v = np.random.randn(n).astype(float, order='F')
    incv = 1
    tau = 1.9

    a_expected, b_expected = apply_householder_right_reference(a, b, v, incv, tau)

    from ctrlsys import _slicot

    a_out = a.copy(order='F')
    b_out = b.copy(order='F')
    _ctrlsys.mb04ny(m, n, v, incv, tau, a_out, b_out)

    np.testing.assert_allclose(a_out, a_expected, rtol=1e-13)
    np.testing.assert_allclose(b_out, b_expected, rtol=1e-13)


def test_mb04ny_tau_zero():
    """
    Test tau=0 case (identity transformation)

    When tau=0, H=I, so matrices should be unchanged.
    Random seed: 999
    """
    np.random.seed(999)

    m, n = 5, 4
    a = np.random.randn(m, 1).astype(float, order='F')
    b = np.random.randn(m, n).astype(float, order='F')
    v = np.random.randn(n).astype(float, order='F')
    incv = 1
    tau = 0.0

    a_expected = a.copy()
    b_expected = b.copy()

    from ctrlsys import _slicot

    a_out = a.copy(order='F')
    b_out = b.copy(order='F')
    _ctrlsys.mb04ny(m, n, v, incv, tau, a_out, b_out)

    np.testing.assert_allclose(a_out, a_expected, rtol=1e-14)
    np.testing.assert_allclose(b_out, b_expected, rtol=1e-14)


def test_mb04ny_orthogonality():
    """
    Mathematical property: H is orthogonal (H'*H = I)

    Verify orthogonality of Householder reflector.
    Random seed: 111
    """
    np.random.seed(111)

    n = 6
    v = np.random.randn(n).astype(float, order='F')
    tau = 2.0 / (1.0 + np.dot(v, v))

    u = np.vstack([[1.0], v.reshape(-1, 1)])
    h = np.eye(n + 1) - tau * (u @ u.T)

    identity = h @ h.T
    np.testing.assert_allclose(identity, np.eye(n + 1), rtol=1e-14, atol=1e-15)


def test_mb04ny_m_zero():
    """
    Edge case: m=0 (empty rows)

    Tests degenerate case with no rows.
    """
    m, n = 0, 4
    a = np.empty((0, 1), dtype=float, order='F')
    b = np.empty((0, n), dtype=float, order='F')
    v = np.random.randn(n).astype(float, order='F')
    incv = 1
    tau = 1.5

    from ctrlsys import _slicot

    a_out = a.copy(order='F')
    b_out = b.copy(order='F')
    _ctrlsys.mb04ny(m, n, v, incv, tau, a_out, b_out)

    assert a_out.shape == (0, 1)
    assert b_out.shape == (0, n)


def test_mb04ny_n_zero():
    """
    Edge case: n=0 (only A column, B is empty)

    Tests 1x1 Householder: A := (1-tau)*A
    Random seed: 222
    """
    np.random.seed(222)

    m, n = 4, 0
    a = np.random.randn(m, 1).astype(float, order='F')
    b = np.empty((m, 0), dtype=float, order='F')
    v = np.empty(0, dtype=float, order='F')
    incv = 1
    tau = 1.5

    a_expected = (1.0 - tau) * a

    from ctrlsys import _slicot

    a_out = a.copy(order='F')
    b_out = b.copy(order='F')
    _ctrlsys.mb04ny(m, n, v, incv, tau, a_out, b_out)

    np.testing.assert_allclose(a_out, a_expected, rtol=1e-14)
    assert b_out.shape == (m, 0)


def test_mb04ny_incv_positive():
    """
    Test positive increment incv=2

    Random seed: 333
    """
    np.random.seed(333)

    m, n = 5, 3
    a = np.random.randn(m, 1).astype(float, order='F')
    b = np.random.randn(m, n).astype(float, order='F')
    # v has length 1 + (n-1)*abs(incv) = 1 + 2*2 = 5
    incv = 2
    v_len = 1 + (n - 1) * abs(incv)
    v = np.random.randn(v_len).astype(float, order='F')
    tau = 1.4

    a_expected, b_expected = apply_householder_right_reference(a, b, v, incv, tau)

    from ctrlsys import _slicot

    a_out = a.copy(order='F')
    b_out = b.copy(order='F')
    _ctrlsys.mb04ny(m, n, v, incv, tau, a_out, b_out)

    np.testing.assert_allclose(a_out, a_expected, rtol=1e-14)
    np.testing.assert_allclose(b_out, b_expected, rtol=1e-14)


def test_mb04ny_incv_negative():
    """
    Test negative increment incv=-1

    Random seed: 444
    """
    np.random.seed(444)

    m, n = 4, 3
    a = np.random.randn(m, 1).astype(float, order='F')
    b = np.random.randn(m, n).astype(float, order='F')
    incv = -1
    v = np.random.randn(n).astype(float, order='F')
    tau = 1.6

    a_expected, b_expected = apply_householder_right_reference(a, b, v, incv, tau)

    from ctrlsys import _slicot

    a_out = a.copy(order='F')
    b_out = b.copy(order='F')
    _ctrlsys.mb04ny(m, n, v, incv, tau, a_out, b_out)

    np.testing.assert_allclose(a_out, a_expected, rtol=1e-14)
    np.testing.assert_allclose(b_out, b_expected, rtol=1e-14)


def test_mb04ny_single_row():
    """
    Edge case: m=1 (single row)

    Random seed: 555
    """
    np.random.seed(555)

    m, n = 1, 5
    a = np.random.randn(m, 1).astype(float, order='F')
    b = np.random.randn(m, n).astype(float, order='F')
    v = np.random.randn(n).astype(float, order='F')
    incv = 1
    tau = 1.7

    a_expected, b_expected = apply_householder_right_reference(a, b, v, incv, tau)

    from ctrlsys import _slicot

    a_out = a.copy(order='F')
    b_out = b.copy(order='F')
    _ctrlsys.mb04ny(m, n, v, incv, tau, a_out, b_out)

    np.testing.assert_allclose(a_out, a_expected, rtol=1e-14)
    np.testing.assert_allclose(b_out, b_expected, rtol=1e-14)
