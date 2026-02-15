"""
Tests for SB03OV - Complex plane rotation construction for Lyapunov solver.

SB03OV constructs a complex plane rotation such that:
    ( conjg(c)  s ) * ( a ) = ( d )
    (    -s     c )   ( b )   ( 0 )

where d is always real and is overwritten on a.
"""
import pytest
import numpy as np
from ctrlsys import sb03ov


def test_sb03ov_basic():
    """
    Test basic functionality with normal inputs.

    Tests that the rotation properly zeroes b and produces real d.
    """
    a_re, a_im = 3.0, 4.0
    b = 5.0
    small = 1e-15

    d, c_re, c_im, s, info = sb03ov(a_re, a_im, b, small)

    assert info == 0
    # d should be norm([a_re, a_im, b])
    expected_d = np.sqrt(a_re**2 + a_im**2 + b**2)
    np.testing.assert_allclose(d, expected_d, rtol=1e-14)
    # c = (a_re/d, a_im/d), s = b/d
    np.testing.assert_allclose(c_re, a_re / expected_d, rtol=1e-14)
    np.testing.assert_allclose(c_im, a_im / expected_d, rtol=1e-14)
    np.testing.assert_allclose(s, b / expected_d, rtol=1e-14)


def test_sb03ov_orthogonality():
    """
    Test orthogonality property: |c|^2 + s^2 = 1.

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    for _ in range(10):
        a_re = np.random.randn()
        a_im = np.random.randn()
        b = np.random.randn()
        small = 1e-15

        d, c_re, c_im, s, info = sb03ov(a_re, a_im, b, small)

        assert info == 0
        # |c|^2 + s^2 = 1
        norm_sq = c_re**2 + c_im**2 + s**2
        np.testing.assert_allclose(norm_sq, 1.0, rtol=1e-14)


def test_sb03ov_transformation_property():
    """
    Test the transformation: conjg(c) * a + s * b = d.

    This verifies the first row of the rotation produces d.
    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    for _ in range(10):
        a_re = np.random.randn()
        a_im = np.random.randn()
        b = np.random.randn()
        small = 1e-15

        d, c_re, c_im, s, info = sb03ov(a_re, a_im, b, small)

        assert info == 0
        # conjg(c) * a + s * b = d (real)
        # conjg(c) = (c_re, -c_im)
        # c * a = (c_re + i*c_im) * (a_re + i*a_im) but we want conjg(c) * a
        # conjg(c) * a = (c_re - i*c_im) * (a_re + i*a_im)
        #              = c_re*a_re + c_im*a_im + i*(c_re*a_im - c_im*a_re)
        result_re = c_re * a_re + c_im * a_im + s * b
        result_im = c_re * a_im - c_im * a_re
        np.testing.assert_allclose(result_re, d, rtol=1e-14)
        np.testing.assert_allclose(result_im, 0.0, atol=1e-14)


def test_sb03ov_annihilation_property():
    """
    Test the annihilation: -s * a + c * b = 0.

    This verifies the second row of the rotation produces 0.
    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    for _ in range(10):
        a_re = np.random.randn()
        a_im = np.random.randn()
        b = np.random.randn()
        small = 1e-15

        d, c_re, c_im, s, info = sb03ov(a_re, a_im, b, small)

        assert info == 0
        # -s * a + c * b should be 0 (both real and imaginary parts)
        # -s * (a_re + i*a_im) + (c_re + i*c_im) * b
        # = (-s*a_re + c_re*b) + i*(-s*a_im + c_im*b)
        result_re = -s * a_re + c_re * b
        result_im = -s * a_im + c_im * b
        np.testing.assert_allclose(result_re, 0.0, atol=1e-14)
        np.testing.assert_allclose(result_im, 0.0, atol=1e-14)


def test_sb03ov_small_norm():
    """
    Test edge case when norm([a, b]) < small.

    When d < small, c = (1, 0), s = 0, and a = (d, 0).
    """
    a_re, a_im = 1e-20, 1e-20
    b = 1e-20
    small = 1e-10

    d, c_re, c_im, s, info = sb03ov(a_re, a_im, b, small)

    assert info == 0
    # For very small inputs with d < small: c = (1, 0), s = 0
    np.testing.assert_allclose(c_re, 1.0, rtol=1e-14)
    np.testing.assert_allclose(c_im, 0.0, atol=1e-14)
    np.testing.assert_allclose(s, 0.0, atol=1e-14)
    # d should be the actual norm if > 0
    expected_d = np.sqrt(a_re**2 + a_im**2 + b**2)
    if expected_d > 0:
        np.testing.assert_allclose(d, expected_d, rtol=1e-10)


def test_sb03ov_zero_input():
    """
    Test edge case with zero inputs.
    """
    a_re, a_im = 0.0, 0.0
    b = 0.0
    small = 1e-15

    d, c_re, c_im, s, info = sb03ov(a_re, a_im, b, small)

    assert info == 0
    # d = 0, c = (1, 0), s = 0
    np.testing.assert_allclose(d, 0.0, atol=1e-14)
    np.testing.assert_allclose(c_re, 1.0, rtol=1e-14)
    np.testing.assert_allclose(c_im, 0.0, atol=1e-14)
    np.testing.assert_allclose(s, 0.0, atol=1e-14)


def test_sb03ov_real_only():
    """
    Test with pure real a (imaginary part = 0).
    """
    a_re, a_im = 3.0, 0.0
    b = 4.0
    small = 1e-15

    d, c_re, c_im, s, info = sb03ov(a_re, a_im, b, small)

    assert info == 0
    expected_d = 5.0  # sqrt(9 + 16)
    np.testing.assert_allclose(d, expected_d, rtol=1e-14)
    np.testing.assert_allclose(c_re, 0.6, rtol=1e-14)  # 3/5
    np.testing.assert_allclose(c_im, 0.0, atol=1e-14)
    np.testing.assert_allclose(s, 0.8, rtol=1e-14)  # 4/5


def test_sb03ov_imaginary_only():
    """
    Test with pure imaginary a (real part = 0).
    """
    a_re, a_im = 0.0, 3.0
    b = 4.0
    small = 1e-15

    d, c_re, c_im, s, info = sb03ov(a_re, a_im, b, small)

    assert info == 0
    expected_d = 5.0  # sqrt(9 + 16)
    np.testing.assert_allclose(d, expected_d, rtol=1e-14)
    np.testing.assert_allclose(c_re, 0.0, atol=1e-14)
    np.testing.assert_allclose(c_im, 0.6, rtol=1e-14)  # 3/5
    np.testing.assert_allclose(s, 0.8, rtol=1e-14)  # 4/5
