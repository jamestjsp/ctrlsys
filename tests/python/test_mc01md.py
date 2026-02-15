"""
Tests for MC01MD - Compute shifted polynomial coefficients using Horner's algorithm.

Given P(x) = p[0] + p[1]*x + ... + p[dp]*x^dp
Computes Q(x) = q[0] + q[1]*(x-alpha) + ... + q[k-1]*(x-alpha)^(k-1) + ...

The relation: q[i] = P^(i)(alpha) / i! for i = 0, 1, ..., k-1
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_mc01md_basic_html_example():
    """
    Test using SLICOT HTML doc example.

    P(x) = 6 + 5x + 4x^2 + 3x^3 + 2x^4 + x^5
    alpha = 2.0, k = 6 (all coefficients)

    Expected shifted polynomial coefficients:
    q[0] = 120 (P(2) = 6+10+16+24+32+32 = 120)
    q[1] = 201 (P'(2) = 5+16+36+64+80 = 201)
    q[2] = 150
    q[3] = 59
    q[4] = 12
    q[5] = 1
    """
    from ctrlsys import mc01md

    dp = 5
    alpha = 2.0
    k = 6
    p = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0], order='F', dtype=float)

    q, info = mc01md(alpha, k, p)

    assert info == 0
    assert len(q) == dp + 1

    expected_q = np.array([120.0, 201.0, 150.0, 59.0, 12.0, 1.0])
    assert_allclose(q[:k], expected_q, rtol=1e-14)


def test_mc01md_partial_coefficients():
    """
    Test computing only first 3 coefficients of shifted polynomial.

    P(x) = 6 + 5x + 4x^2 + 3x^3 + 2x^4 + x^5
    alpha = 2.0, k = 3

    Only q[0], q[1], q[2] should be meaningful.
    """
    from ctrlsys import mc01md

    dp = 5
    alpha = 2.0
    k = 3
    p = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0], order='F', dtype=float)

    q, info = mc01md(alpha, k, p)

    assert info == 0
    assert_allclose(q[0], 120.0, rtol=1e-14)
    assert_allclose(q[1], 201.0, rtol=1e-14)
    assert_allclose(q[2], 150.0, rtol=1e-14)


def test_mc01md_zero_alpha():
    """
    Test with alpha = 0 (no shift).

    When alpha = 0, P(x) in powers of x = P(x) in powers of (x-0).
    Output should equal input.

    P(x) = 1 + 2x + 3x^2
    """
    from ctrlsys import mc01md

    dp = 2
    alpha = 0.0
    k = 3
    p = np.array([1.0, 2.0, 3.0], order='F', dtype=float)

    q, info = mc01md(alpha, k, p)

    assert info == 0
    assert_allclose(q, p, rtol=1e-14)


def test_mc01md_constant_polynomial():
    """
    Test with constant polynomial (dp = 0).

    P(x) = 5 (constant)
    Shifted: P(x) = 5 (still constant, q[0] = 5)
    """
    from ctrlsys import mc01md

    dp = 0
    alpha = 3.0
    k = 1
    p = np.array([5.0], order='F', dtype=float)

    q, info = mc01md(alpha, k, p)

    assert info == 0
    assert_allclose(q[0], 5.0, rtol=1e-14)


def test_mc01md_linear_polynomial():
    """
    Test with linear polynomial.

    P(x) = 2 + 3x
    alpha = 1.0

    P(x) in powers of (x-1):
    P(x) = P(1) + P'(1)*(x-1) = 5 + 3*(x-1)
    q[0] = 5, q[1] = 3
    """
    from ctrlsys import mc01md

    dp = 1
    alpha = 1.0
    k = 2
    p = np.array([2.0, 3.0], order='F', dtype=float)

    q, info = mc01md(alpha, k, p)

    assert info == 0
    assert_allclose(q[0], 5.0, rtol=1e-14)
    assert_allclose(q[1], 3.0, rtol=1e-14)


def test_mc01md_quadratic_taylor():
    """
    Test quadratic polynomial Taylor expansion property.

    P(x) = 1 + x + x^2
    alpha = 2.0

    Taylor expansion at alpha=2:
    P(2) = 1 + 2 + 4 = 7
    P'(2) = 1 + 2*2 = 5
    P''(2)/2! = 2/2 = 1

    So: P(x) = 7 + 5*(x-2) + 1*(x-2)^2

    Random seed: not used (deterministic test data)
    """
    from ctrlsys import mc01md

    dp = 2
    alpha = 2.0
    k = 3
    p = np.array([1.0, 1.0, 1.0], order='F', dtype=float)

    q, info = mc01md(alpha, k, p)

    assert info == 0
    assert_allclose(q[0], 7.0, rtol=1e-14)
    assert_allclose(q[1], 5.0, rtol=1e-14)
    assert_allclose(q[2], 1.0, rtol=1e-14)


def test_mc01md_single_coefficient():
    """
    Test computing only the constant term (k=1).

    P(x) = 1 + 2x + 3x^2 + 4x^3
    alpha = -1.0

    q[0] = P(-1) = 1 - 2 + 3 - 4 = -2
    """
    from ctrlsys import mc01md

    dp = 3
    alpha = -1.0
    k = 1
    p = np.array([1.0, 2.0, 3.0, 4.0], order='F', dtype=float)

    q, info = mc01md(alpha, k, p)

    assert info == 0
    assert_allclose(q[0], -2.0, rtol=1e-14)


def test_mc01md_error_negative_dp():
    """
    Test error for dp < 0 (via insufficient array size).

    Note: Python wrapper infers dp from array size.
    Empty array means dp = -1 which is invalid.
    """
    from ctrlsys import mc01md

    alpha = 1.0
    k = 1
    p = np.array([], order='F', dtype=float)

    q, info = mc01md(alpha, k, p)

    assert info == -1


def test_mc01md_error_invalid_k_zero():
    """
    Test error for k <= 0.

    K is the 3rd argument so error code is -3.
    """
    from ctrlsys import mc01md

    alpha = 1.0
    k = 0
    p = np.array([1.0, 2.0], order='F', dtype=float)

    q, info = mc01md(alpha, k, p)

    assert info == -3


def test_mc01md_error_k_too_large():
    """
    Test error for k > dp + 1.

    dp = 2, so max k = 3. k = 4 is invalid.
    K is the 3rd argument so error code is -3.
    """
    from ctrlsys import mc01md

    alpha = 1.0
    k = 4
    p = np.array([1.0, 2.0, 3.0], order='F', dtype=float)

    q, info = mc01md(alpha, k, p)

    assert info == -3


def test_mc01md_taylor_property():
    """
    Validate mathematical property: q[i] = P^(i)(alpha) / i!

    For any polynomial, the shifted coefficients are Taylor coefficients.

    P(x) = 2 - 3x + x^2 + 4x^3
    alpha = 0.5

    P(0.5) = 2 - 1.5 + 0.25 + 0.5 = 1.25
    P'(x) = -3 + 2x + 12x^2 -> P'(0.5) = -3 + 1 + 3 = 1
    P''(x) = 2 + 24x -> P''(0.5) = 2 + 12 = 14
    P'''(x) = 24 -> P'''(0.5) = 24

    q[0] = 1.25
    q[1] = 1
    q[2] = 14/2 = 7
    q[3] = 24/6 = 4

    Random seed: not used (deterministic test data)
    """
    from ctrlsys import mc01md

    dp = 3
    alpha = 0.5
    k = 4
    p = np.array([2.0, -3.0, 1.0, 4.0], order='F', dtype=float)

    q, info = mc01md(alpha, k, p)

    assert info == 0
    assert_allclose(q[0], 1.25, rtol=1e-14)
    assert_allclose(q[1], 1.0, rtol=1e-14)
    assert_allclose(q[2], 7.0, rtol=1e-14)
    assert_allclose(q[3], 4.0, rtol=1e-14)


def test_mc01md_polynomial_evaluation_invariant():
    """
    Validate invariant: P(x) evaluated at any x should equal Q(x-alpha).

    For P(x) and its shifted form Q centered at alpha,
    P(x) = Q(x-alpha) for all x.

    Test at multiple points.

    Random seed: 42 (for reproducibility)
    """
    from ctrlsys import mc01md

    np.random.seed(42)

    dp = 4
    alpha = 1.5
    k = dp + 1
    p = np.array([1.0, -2.0, 3.0, -1.0, 0.5], order='F', dtype=float)

    q, info = mc01md(alpha, k, p)
    assert info == 0

    test_points = np.array([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0])

    for x in test_points:
        p_at_x = np.polyval(p[::-1], x)
        q_at_x_minus_alpha = np.polyval(q[::-1], x - alpha)
        assert_allclose(p_at_x, q_at_x_minus_alpha, rtol=1e-14)


def test_mc01md_random_polynomial():
    """
    Test with random polynomial and verify Taylor property.

    Random seed: 123 (for reproducibility)
    """
    from ctrlsys import mc01md

    np.random.seed(123)

    dp = 5
    alpha = np.random.uniform(-2, 2)
    k = dp + 1
    p = np.random.randn(dp + 1).astype(float)
    p_fortran = np.array(p, order='F', dtype=float)

    q, info = mc01md(alpha, k, p_fortran)
    assert info == 0

    x_test = np.linspace(-3, 3, 20)
    for x in x_test:
        p_val = np.polyval(p[::-1], x)
        q_val = np.polyval(q[::-1], x - alpha)
        assert_allclose(p_val, q_val, rtol=1e-12, atol=1e-14)
