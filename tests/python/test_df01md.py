"""Tests for DF01MD - Sine or cosine transform of a real signal."""

import numpy as np
import pytest
from ctrlsys import df01md


"""Basic functionality tests from SLICOT HTML documentation."""


def test_cosine_transform_doc_example():
    """
    Test cosine transform using HTML doc example.

    Input: N=17, DT=1.0, SICO='C'
    A = [-0.1862, 0.1288, 0.3948, 0.0671, 0.6788, -0.2417, 0.1861, 0.8875,
         0.7254, 0.9380, 0.5815, -0.2682, 0.4904, 0.9312, -0.9599, -0.3116, 0.8743]

    Expected output (from HTML doc):
    A = [28.0536, 3.3726, -20.8158, 6.0566, 5.7317, -3.9347, -12.8074, -6.8780,
         16.2892, -17.0788, 21.7836, -20.8203, -7.3277, -2.5325, -0.3636, 7.8792, 11.0048]
    """
    a = np.array([-0.1862, 0.1288, 0.3948, 0.0671, 0.6788, -0.2417, 0.1861, 0.8875,
                  0.7254, 0.9380, 0.5815, -0.2682, 0.4904, 0.9312, -0.9599, -0.3116, 0.8743],
                 dtype=float, order='F')
    dt = 1.0

    a_expected = np.array([28.0536, 3.3726, -20.8158, 6.0566, 5.7317, -3.9347, -12.8074, -6.8780,
                           16.2892, -17.0788, 21.7836, -20.8203, -7.3277, -2.5325, -0.3636, 7.8792, 11.0048],
                          dtype=float)

    a_out, info = df01md('C', dt, a)

    assert info == 0
    np.testing.assert_allclose(a_out, a_expected, rtol=1e-3, atol=1e-4)


def test_sine_transform_basic():
    """
    Test sine transform with simple signal.

    For sine transform, first and last coefficients are always 0.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    n = 17
    a = np.random.randn(n).astype(float, order='F')
    dt = 0.1

    a_out, info = df01md('S', dt, a)

    assert info == 0
    np.testing.assert_allclose(a_out[0], 0.0, atol=1e-14)
    np.testing.assert_allclose(a_out[-1], 0.0, atol=1e-14)


"""Mathematical property tests for sine/cosine transforms."""


def test_cosine_transform_linearity():
    """
    Validate cosine transform linearity: C(a*x + b*y) = a*C(x) + b*C(y).

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    n = 17
    alpha, beta = 2.5, -1.3
    dt = 1.0

    x = np.random.randn(n).astype(float, order='F')
    y = np.random.randn(n).astype(float, order='F')
    z = alpha * x + beta * y

    cx, info1 = df01md('C', dt, x.copy())
    cy, info2 = df01md('C', dt, y.copy())
    cz, info3 = df01md('C', dt, z.copy())

    assert info1 == 0
    assert info2 == 0
    assert info3 == 0

    np.testing.assert_allclose(cz, alpha * cx + beta * cy, rtol=1e-12)


def test_sine_transform_linearity():
    """
    Validate sine transform linearity: S(a*x + b*y) = a*S(x) + b*S(y).

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    n = 17
    alpha, beta = 1.5, -0.8
    dt = 1.0

    x = np.random.randn(n).astype(float, order='F')
    y = np.random.randn(n).astype(float, order='F')
    z = alpha * x + beta * y

    sx, info1 = df01md('S', dt, x.copy())
    sy, info2 = df01md('S', dt, y.copy())
    sz, info3 = df01md('S', dt, z.copy())

    assert info1 == 0
    assert info2 == 0
    assert info3 == 0

    np.testing.assert_allclose(sz, alpha * sx + beta * sy, rtol=1e-12)


def test_sine_transform_boundary_conditions():
    """
    Validate sine transform boundary conditions.

    For sine transform:
    - S_1 = 0 (first coefficient is zero)
    - S_N = 0 (last coefficient is zero)

    Random seed: 789 (for reproducibility)
    """
    np.random.seed(789)
    n = 33
    a = np.random.randn(n).astype(float, order='F')
    dt = 0.5

    s, info = df01md('S', dt, a)

    assert info == 0
    np.testing.assert_allclose(s[0], 0.0, atol=1e-14)
    np.testing.assert_allclose(s[-1], 0.0, atol=1e-14)


def test_cosine_transform_first_last_relation():
    """
    Validate cosine transform first/last coefficient formulas.

    From documentation:
    S_1 = 2*DT*[D(1) + A0]
    S_N = 2*DT*[D(1) - A0]

    Where A0 = 2*SUM_{i=1}^{(N-1)/2} A(2i)

    This implies: S_1 + S_N = 4*DT*D(1) and S_1 - S_N = 4*DT*A0

    Random seed: 100 (for reproducibility)
    """
    np.random.seed(100)
    n = 17
    a = np.random.randn(n).astype(float, order='F')
    dt = 1.0

    a0 = 2.0 * np.sum(a[1::2])

    c, info = df01md('C', dt, a.copy())
    assert info == 0

    np.testing.assert_allclose(c[0] - c[-1], 4 * dt * a0, rtol=1e-12)


def test_zero_signal():
    """Test that zero signal gives zero output for both transforms."""
    n = 17
    a_zero = np.zeros(n, dtype=float, order='F')
    dt = 1.0

    c, info_c = df01md('C', dt, a_zero.copy())
    s, info_s = df01md('S', dt, a_zero.copy())

    assert info_c == 0
    assert info_s == 0
    np.testing.assert_allclose(c, np.zeros(n), atol=1e-15)
    np.testing.assert_allclose(s, np.zeros(n), atol=1e-15)


def test_dt_scaling():
    """
    Validate that output scales linearly with DT.

    C(a, 2*dt) = 2 * C(a, dt)

    Random seed: 200 (for reproducibility)
    """
    np.random.seed(200)
    n = 17
    a = np.random.randn(n).astype(float, order='F')
    dt = 0.5

    c1, info1 = df01md('C', dt, a.copy())
    c2, info2 = df01md('C', 2 * dt, a.copy())

    assert info1 == 0
    assert info2 == 0
    np.testing.assert_allclose(c2, 2 * c1, rtol=1e-14)


"""Edge case tests."""


def test_minimum_n():
    """Test minimum valid N=5 (power of 2 plus 1 = 4+1=5)."""
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float, order='F')
    dt = 1.0

    c, info = df01md('C', dt, a)
    assert info == 0
    assert len(c) == 5

    s, info_s = df01md('S', dt, a.copy())
    assert info_s == 0
    assert len(s) == 5
    np.testing.assert_allclose(s[0], 0.0, atol=1e-14)
    np.testing.assert_allclose(s[-1], 0.0, atol=1e-14)


def test_n_9():
    """Test N=9 (power of 2 plus 1 = 8+1=9)."""
    np.random.seed(300)
    a = np.random.randn(9).astype(float, order='F')
    dt = 1.0

    c, info = df01md('C', dt, a)
    assert info == 0
    assert len(c) == 9


def test_n_33():
    """Test N=33 (power of 2 plus 1 = 32+1=33)."""
    np.random.seed(301)
    a = np.random.randn(33).astype(float, order='F')
    dt = 0.1

    c, info_c = df01md('C', dt, a.copy())
    s, info_s = df01md('S', dt, a.copy())

    assert info_c == 0
    assert info_s == 0
    assert len(c) == 33
    assert len(s) == 33


def test_n_65():
    """Test N=65 (power of 2 plus 1 = 64+1=65)."""
    np.random.seed(302)
    a = np.random.randn(65).astype(float, order='F')
    dt = 0.01

    c, info = df01md('C', dt, a)
    assert info == 0
    assert len(c) == 65


def test_n_129():
    """Test N=129 (power of 2 plus 1 = 128+1=129, max from example)."""
    np.random.seed(303)
    a = np.random.randn(129).astype(float, order='F')
    dt = 0.001

    c, info = df01md('C', dt, a)
    assert info == 0
    assert len(c) == 129


"""Error handling tests."""


def test_invalid_sico():
    """Test with invalid SICO parameter."""
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float, order='F')
    dt = 1.0

    a_out, info = df01md('X', dt, a)
    assert info == -1


def test_n_less_than_5():
    """Test with N < 5 (minimum is 5)."""
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=float, order='F')
    dt = 1.0

    a_out, info = df01md('C', dt, a)
    assert info == -2


def test_n_not_power_of_2_plus_1():
    """Test with N not equal to power of 2 plus 1."""
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float, order='F')
    dt = 1.0

    a_out, info = df01md('C', dt, a)
    assert info == -2


def test_n_10():
    """Test with N=10 (not power of 2 plus 1)."""
    a = np.ones(10, dtype=float, order='F')
    dt = 1.0

    a_out, info = df01md('C', dt, a)
    assert info == -2


def test_lowercase_sico():
    """Test that lowercase SICO parameters work."""
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float, order='F')
    dt = 1.0

    c, info_c = df01md('c', dt, a.copy())
    s, info_s = df01md('s', dt, a.copy())

    assert info_c == 0
    assert info_s == 0
