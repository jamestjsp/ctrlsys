"""
Tests for MC01ND - Evaluate real polynomial at complex point using Horner's algorithm.

Given P(x) = p[0] + p[1]*x + p[2]*x^2 + ... + p[dp]*x^dp
Computes P(x0) where x0 = xr + xi*j is complex.

Uses Horner's recursion:
  q[dp] = p[dp]
  q[i] = x0*q[i+1] + p[i] for i = dp-1, ..., 0
Result: P(x0) = q[0]
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_mc01nd_basic_html_example():
    """
    Test using SLICOT HTML doc example.

    P(x) = 5 + 3x - x^2 + 2x^3 + x^4
    x0 = -1.56 + 0.29j

    Expected: P(x0) = -4.1337 + 1.7088j
    Tolerance based on HTML 4-decimal display precision.
    """
    from ctrlsys import mc01nd

    dp = 4
    xr = -1.56
    xi = 0.29
    p = np.array([5.0, 3.0, -1.0, 2.0, 1.0], order='F', dtype=float)

    vr, vi, info = mc01nd(xr, xi, p)

    assert info == 0
    assert_allclose(vr, -4.1337, rtol=1e-3, atol=1e-4)
    assert_allclose(vi, 1.7088, rtol=1e-3, atol=1e-4)


def test_mc01nd_real_evaluation():
    """
    Test polynomial evaluation at real point (xi = 0).

    P(x) = 1 + 2x + 3x^2
    x0 = 2.0 (real)

    P(2) = 1 + 4 + 12 = 17

    Expected: vr=17.0, vi=0.0
    """
    from ctrlsys import mc01nd

    dp = 2
    xr = 2.0
    xi = 0.0
    p = np.array([1.0, 2.0, 3.0], order='F', dtype=float)

    vr, vi, info = mc01nd(xr, xi, p)

    assert info == 0
    assert_allclose(vr, 17.0, rtol=1e-14)
    assert_allclose(vi, 0.0, atol=1e-14)


def test_mc01nd_purely_imaginary_point():
    """
    Test evaluation at purely imaginary point.

    P(x) = 1 + x + x^2
    x0 = j (purely imaginary: xr=0, xi=1)

    P(j) = 1 + j + j^2 = 1 + j - 1 = j

    Expected: vr=0.0, vi=1.0
    """
    from ctrlsys import mc01nd

    dp = 2
    xr = 0.0
    xi = 1.0
    p = np.array([1.0, 1.0, 1.0], order='F', dtype=float)

    vr, vi, info = mc01nd(xr, xi, p)

    assert info == 0
    assert_allclose(vr, 0.0, atol=1e-14)
    assert_allclose(vi, 1.0, rtol=1e-14)


def test_mc01nd_constant_polynomial():
    """
    Test constant polynomial (dp = 0).

    P(x) = 7.5 (constant)
    x0 = 3.0 + 4.0j

    P(any) = 7.5

    Expected: vr=7.5, vi=0.0
    """
    from ctrlsys import mc01nd

    dp = 0
    xr = 3.0
    xi = 4.0
    p = np.array([7.5], order='F', dtype=float)

    vr, vi, info = mc01nd(xr, xi, p)

    assert info == 0
    assert_allclose(vr, 7.5, rtol=1e-14)
    assert_allclose(vi, 0.0, atol=1e-14)


def test_mc01nd_linear_polynomial():
    """
    Test linear polynomial.

    P(x) = 2 + 3x
    x0 = 1 + 2j

    P(1+2j) = 2 + 3*(1+2j) = 2 + 3 + 6j = 5 + 6j

    Expected: vr=5.0, vi=6.0
    """
    from ctrlsys import mc01nd

    dp = 1
    xr = 1.0
    xi = 2.0
    p = np.array([2.0, 3.0], order='F', dtype=float)

    vr, vi, info = mc01nd(xr, xi, p)

    assert info == 0
    assert_allclose(vr, 5.0, rtol=1e-14)
    assert_allclose(vi, 6.0, rtol=1e-14)


def test_mc01nd_quadratic_complex():
    """
    Test quadratic polynomial at complex point.

    P(x) = 1 + 2x + x^2
    x0 = 1 + j

    x0^2 = (1+j)^2 = 1 + 2j - 1 = 2j
    P(1+j) = 1 + 2*(1+j) + 2j = 1 + 2 + 2j + 2j = 3 + 4j

    Expected: vr=3.0, vi=4.0
    """
    from ctrlsys import mc01nd

    dp = 2
    xr = 1.0
    xi = 1.0
    p = np.array([1.0, 2.0, 1.0], order='F', dtype=float)

    vr, vi, info = mc01nd(xr, xi, p)

    assert info == 0
    assert_allclose(vr, 3.0, rtol=1e-14)
    assert_allclose(vi, 4.0, rtol=1e-14)


def test_mc01nd_error_negative_dp():
    """
    Test error for dp < 0 (via empty array).

    Python wrapper infers dp from array size.
    Empty array means dp = -1 which is invalid.
    """
    from ctrlsys import mc01nd

    xr = 1.0
    xi = 0.0
    p = np.array([], order='F', dtype=float)

    vr, vi, info = mc01nd(xr, xi, p)

    assert info == -1


def test_mc01nd_numpy_comparison():
    """
    Validate against numpy.polyval for complex evaluation.

    P(x) = 2 - 3x + x^2 + 4x^3
    x0 = -0.5 + 1.5j

    Random seed: not used (deterministic test data)
    """
    from ctrlsys import mc01nd

    dp = 3
    xr = -0.5
    xi = 1.5
    p = np.array([2.0, -3.0, 1.0, 4.0], order='F', dtype=float)

    vr, vi, info = mc01nd(xr, xi, p)
    assert info == 0

    x0 = complex(xr, xi)
    expected = np.polyval(p[::-1], x0)

    assert_allclose(vr, expected.real, rtol=1e-14)
    assert_allclose(vi, expected.imag, rtol=1e-14)


def test_mc01nd_horner_property():
    """
    Validate Horner's algorithm correctness.

    For P(x) = p0 + p1*x + p2*x^2 + p3*x^3
    Horner's computes: ((p3*x + p2)*x + p1)*x + p0

    Test with specific values to verify step-by-step.

    P(x) = 1 + 2x + 3x^2 + 4x^3
    x0 = 2 + j

    Manual Horner:
      q3 = 4
      q2 = 4*(2+j) + 3 = 8 + 4j + 3 = 11 + 4j
      q1 = (11+4j)*(2+j) + 2 = 22 + 11j + 8j - 4 + 2 = 20 + 19j
      q0 = (20+19j)*(2+j) + 1 = 40 + 20j + 38j - 19 + 1 = 22 + 58j

    Expected: P(2+j) = 22 + 58j
    """
    from ctrlsys import mc01nd

    dp = 3
    xr = 2.0
    xi = 1.0
    p = np.array([1.0, 2.0, 3.0, 4.0], order='F', dtype=float)

    vr, vi, info = mc01nd(xr, xi, p)

    assert info == 0
    assert_allclose(vr, 22.0, rtol=1e-14)
    assert_allclose(vi, 58.0, rtol=1e-14)


def test_mc01nd_random_polynomial_numpy_cross_validation():
    """
    Cross-validate with numpy for random polynomial.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mc01nd

    np.random.seed(42)

    dp = 6
    p = np.random.randn(dp + 1).astype(float)
    p_f = np.array(p, order='F', dtype=float)

    xr = np.random.uniform(-2, 2)
    xi = np.random.uniform(-2, 2)

    vr, vi, info = mc01nd(xr, xi, p_f)
    assert info == 0

    x0 = complex(xr, xi)
    expected = np.polyval(p[::-1], x0)

    assert_allclose(vr, expected.real, rtol=1e-13)
    assert_allclose(vi, expected.imag, rtol=1e-13)


def test_mc01nd_roots_polynomial():
    """
    Test polynomial at its roots.

    P(x) = x^2 + 1 has roots at x = +j and x = -j

    P(j) = j^2 + 1 = -1 + 1 = 0
    P(-j) = (-j)^2 + 1 = -1 + 1 = 0
    """
    from ctrlsys import mc01nd

    p = np.array([1.0, 0.0, 1.0], order='F', dtype=float)

    vr1, vi1, info1 = mc01nd(0.0, 1.0, p)
    assert info1 == 0
    assert_allclose(vr1, 0.0, atol=1e-14)
    assert_allclose(vi1, 0.0, atol=1e-14)

    vr2, vi2, info2 = mc01nd(0.0, -1.0, p)
    assert info2 == 0
    assert_allclose(vr2, 0.0, atol=1e-14)
    assert_allclose(vi2, 0.0, atol=1e-14)


def test_mc01nd_conjugate_symmetry():
    """
    Validate mathematical property: P(conj(x0)) = conj(P(x0)) for real polynomial.

    For real polynomial P, evaluating at conjugate points gives conjugate results.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mc01nd

    np.random.seed(123)

    dp = 5
    p = np.random.randn(dp + 1).astype(float)
    p_f = np.array(p, order='F', dtype=float)

    xr = 1.5
    xi = 2.3

    vr1, vi1, info1 = mc01nd(xr, xi, p_f)
    assert info1 == 0

    vr2, vi2, info2 = mc01nd(xr, -xi, p_f)
    assert info2 == 0

    assert_allclose(vr1, vr2, rtol=1e-14)
    assert_allclose(vi1, -vi2, rtol=1e-14)


def test_mc01nd_high_degree_polynomial():
    """
    Test higher degree polynomial.

    P(x) = 1 + x + x^2 + x^3 + x^4 + x^5 + x^6
    This is (x^7 - 1)/(x - 1) for x != 1

    x0 = 0.5 + 0.5j

    Compute via geometric series formula and compare.

    Random seed: not used (deterministic test data)
    """
    from ctrlsys import mc01nd

    dp = 6
    p = np.ones(dp + 1, dtype=float)
    p_f = np.array(p, order='F', dtype=float)

    xr = 0.5
    xi = 0.5
    x0 = complex(xr, xi)

    vr, vi, info = mc01nd(xr, xi, p_f)
    assert info == 0

    if abs(x0 - 1.0) > 1e-10:
        expected = (x0**7 - 1) / (x0 - 1)
    else:
        expected = complex(dp + 1, 0)

    assert_allclose(vr, expected.real, rtol=1e-13)
    assert_allclose(vi, expected.imag, rtol=1e-13)


def test_mc01nd_negative_coefficients():
    """
    Test polynomial with negative coefficients.

    P(x) = -1 - 2x + 3x^2 - 4x^3
    x0 = 1 - j

    Manual computation:
    x0^2 = (1-j)^2 = 1 - 2j - 1 = -2j
    x0^3 = x0^2 * x0 = -2j*(1-j) = -2j + 2j^2 = -2 - 2j

    P(x0) = -1 + (-2)*(1-j) + 3*(-2j) + (-4)*(-2-2j)
          = -1 - 2 + 2j - 6j + 8 + 8j
          = 5 + 4j
    """
    from ctrlsys import mc01nd

    dp = 3
    xr = 1.0
    xi = -1.0
    p = np.array([-1.0, -2.0, 3.0, -4.0], order='F', dtype=float)

    vr, vi, info = mc01nd(xr, xi, p)

    assert info == 0
    assert_allclose(vr, 5.0, rtol=1e-14)
    assert_allclose(vi, 4.0, rtol=1e-14)
