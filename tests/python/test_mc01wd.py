"""
Tests for mc01wd - polynomial quotient and remainder for quadratic divisor.

Computes Q(x) and R(x) such that P(x) = B(x) * Q(x) + R(x)
where B(x) = u1 + u2*x + x^2.
"""

import numpy as np
import pytest
from ctrlsys import mc01wd


def test_mc01wd_html_example():
    """
    Test from SLICOT HTML documentation example.

    P(x) = 0.62 + 1.10*x + 1.64*x^2 + 1.88*x^3 + 2.12*x^4 + 1.70*x^5 + 1.00*x^6
    B(x) = 0.60 + 0.80*x + x^2 (u1=0.60, u2=0.80)

    Expected quotient: Q(x) = 0.6 + 0.7*x + 0.8*x^2 + 0.9*x^3 + 1.0*x^4
    Expected remainder: R(x) = 0.26 + 0.2*x

    R(x) = q(1) + q(2)*(u2 + x) = q(1) + 0.8*q(2) + q(2)*x
    So q(2) = 0.2, q(1) = 0.26 - 0.8*0.2 = 0.10
    """
    p = np.array([0.62, 1.10, 1.64, 1.88, 2.12, 1.70, 1.00], dtype=float)
    u1 = 0.60
    u2 = 0.80

    q, info = mc01wd(p, u1, u2)

    assert info == 0
    assert len(q) == 7

    expected_q1 = 0.10
    expected_q2 = 0.20
    expected_quotient = np.array([0.6, 0.7, 0.8, 0.9, 1.0])

    np.testing.assert_allclose(q[0], expected_q1, rtol=1e-10)
    np.testing.assert_allclose(q[1], expected_q2, rtol=1e-10)
    np.testing.assert_allclose(q[2:], expected_quotient, rtol=1e-10)


def test_mc01wd_verify_division():
    """
    Verify polynomial division property: P(x) = B(x)*Q(x) + R(x).

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    p = np.array([0.62, 1.10, 1.64, 1.88, 2.12, 1.70, 1.00], dtype=float)
    u1 = 0.60
    u2 = 0.80

    q_out, info = mc01wd(p, u1, u2)
    assert info == 0

    q1, q2 = q_out[0], q_out[1]
    quotient = q_out[2:]

    for x in np.linspace(-2, 2, 20):
        b_x = u1 + u2 * x + x**2
        q_x = np.polyval(quotient[::-1], x)
        r_x = q1 + q2 * (u2 + x)
        p_x = np.polyval(p[::-1], x)

        np.testing.assert_allclose(b_x * q_x + r_x, p_x, rtol=1e-13)


def test_mc01wd_degree_zero():
    """Test polynomial of degree 0: P(x) = c (constant)."""
    p = np.array([3.5], dtype=float)
    u1 = 1.0
    u2 = 2.0

    q, info = mc01wd(p, u1, u2)

    assert info == 0
    assert len(q) == 1
    np.testing.assert_allclose(q[0], 3.5, rtol=1e-14)


def test_mc01wd_degree_one():
    """Test polynomial of degree 1: P(x) = a + b*x."""
    p = np.array([2.0, 3.0], dtype=float)
    u1 = 0.5
    u2 = 1.0

    q, info = mc01wd(p, u1, u2)

    assert info == 0
    assert len(q) == 2
    np.testing.assert_allclose(q[1], 3.0, rtol=1e-14)
    np.testing.assert_allclose(q[0], p[0] - u2 * p[1], rtol=1e-14)


def test_mc01wd_exact_division():
    """
    Test exact division where remainder is zero.

    P(x) = (x^2 + 2x + 1) * (x + 1) = x^3 + 3x^2 + 3x + 1
    B(x) = 1 + 2x + x^2 (u1=1, u2=2)
    Q(x) = 1 + x
    R(x) = 0
    """
    p = np.array([1.0, 3.0, 3.0, 1.0], dtype=float)
    u1 = 1.0
    u2 = 2.0

    q, info = mc01wd(p, u1, u2)

    assert info == 0
    np.testing.assert_allclose(q[0], 0.0, atol=1e-14)
    np.testing.assert_allclose(q[1], 0.0, atol=1e-14)
    np.testing.assert_allclose(q[2:], [1.0, 1.0], rtol=1e-14)


def test_mc01wd_random_verification():
    """
    Random polynomial verification: P(x) = B(x)*Q(x) + R(x).

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    dp = 5
    p = np.random.randn(dp + 1)
    u1 = np.random.randn()
    u2 = np.random.randn()

    q_out, info = mc01wd(p, u1, u2)
    assert info == 0

    q1, q2 = q_out[0], q_out[1]
    quotient = q_out[2:]

    x_test = np.linspace(-3, 3, 50)
    for x in x_test:
        b_x = u1 + u2 * x + x**2
        q_x = np.polyval(quotient[::-1], x)
        r_x = q1 + q2 * (u2 + x)
        p_x = np.polyval(p[::-1], x)

        np.testing.assert_allclose(b_x * q_x + r_x, p_x, rtol=1e-12)
