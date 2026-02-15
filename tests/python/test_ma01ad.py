"""
Tests for MA01AD - Complex square root computation in real arithmetic.

MA01AD computes the complex square root YR + i*YI of a complex number XR + i*XI.
The result satisfies: YR >= 0 and SIGN(YI) = SIGN(XI).

Test data sources:
- Mathematical properties of square root
- Known special cases
"""

import numpy as np
import pytest

from ctrlsys import ma01ad


def test_ma01ad_positive_real():
    """
    Test square root of positive real number.
    sqrt(4 + 0i) = 2 + 0i
    """
    xr, xi = 4.0, 0.0
    yr, yi = ma01ad(xr, xi)

    assert yr == pytest.approx(2.0, rel=1e-14)
    assert yi == pytest.approx(0.0, abs=1e-15)


def test_ma01ad_negative_real():
    """
    Test square root of negative real number.
    sqrt(-4 + 0i) = 0 + 2i
    """
    xr, xi = -4.0, 0.0
    yr, yi = ma01ad(xr, xi)

    assert yr == pytest.approx(0.0, abs=1e-15)
    assert yi == pytest.approx(2.0, rel=1e-14)


def test_ma01ad_pure_imaginary_positive():
    """
    Test square root of pure imaginary (positive).
    sqrt(0 + 2i) = 1 + i  (since (1+i)^2 = 1 + 2i - 1 = 2i)
    """
    xr, xi = 0.0, 2.0
    yr, yi = ma01ad(xr, xi)

    assert yr == pytest.approx(1.0, rel=1e-14)
    assert yi == pytest.approx(1.0, rel=1e-14)


def test_ma01ad_pure_imaginary_negative():
    """
    Test square root of pure imaginary (negative).
    sqrt(0 - 2i) = 1 - i  (since (1-i)^2 = 1 - 2i - 1 = -2i)
    """
    xr, xi = 0.0, -2.0
    yr, yi = ma01ad(xr, xi)

    assert yr == pytest.approx(1.0, rel=1e-14)
    assert yi == pytest.approx(-1.0, rel=1e-14)


def test_ma01ad_complex_first_quadrant():
    """
    Test square root of complex number in first quadrant.
    sqrt(3 + 4i) = 2 + i  (since (2+i)^2 = 4 + 4i - 1 = 3 + 4i)
    """
    xr, xi = 3.0, 4.0
    yr, yi = ma01ad(xr, xi)

    assert yr == pytest.approx(2.0, rel=1e-14)
    assert yi == pytest.approx(1.0, rel=1e-14)


def test_ma01ad_complex_fourth_quadrant():
    """
    Test square root of complex number in fourth quadrant.
    sqrt(3 - 4i) = 2 - i  (since (2-i)^2 = 4 - 4i - 1 = 3 - 4i)
    """
    xr, xi = 3.0, -4.0
    yr, yi = ma01ad(xr, xi)

    assert yr == pytest.approx(2.0, rel=1e-14)
    assert yi == pytest.approx(-1.0, rel=1e-14)


def test_ma01ad_squaring_property():
    """
    Mathematical property: (yr + i*yi)^2 = xr + i*xi

    Expanding: (yr + i*yi)^2 = yr^2 - yi^2 + 2*yr*yi*i
    So: yr^2 - yi^2 = xr and 2*yr*yi = xi

    Random seed: 42 (for reproducibility)
    """
    np.random.seed(42)
    for _ in range(10):
        xr = np.random.randn() * 100
        xi = np.random.randn() * 100

        yr, yi = ma01ad(xr, xi)

        # Verify squaring gives back original
        result_real = yr * yr - yi * yi
        result_imag = 2 * yr * yi

        np.testing.assert_allclose(result_real, xr, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(result_imag, xi, rtol=1e-13, atol=1e-14)


def test_ma01ad_yr_nonnegative():
    """
    Property: YR >= 0 for all inputs.

    Random seed: 123 (for reproducibility)
    """
    np.random.seed(123)
    for _ in range(20):
        xr = np.random.randn() * 100
        xi = np.random.randn() * 100

        yr, yi = ma01ad(xr, xi)

        assert yr >= 0.0, f"YR should be non-negative, got {yr}"


def test_ma01ad_sign_preservation():
    """
    Property: SIGN(YI) = SIGN(XI).

    Random seed: 456 (for reproducibility)
    """
    np.random.seed(456)
    for _ in range(20):
        xr = np.random.randn() * 100
        xi = np.random.randn() * 100

        yr, yi = ma01ad(xr, xi)

        if xi > 0:
            assert yi >= 0, f"YI should be non-negative when XI > 0"
        elif xi < 0:
            assert yi <= 0, f"YI should be non-positive when XI < 0"


def test_ma01ad_zero():
    """
    Test square root of zero.
    sqrt(0 + 0i) = 0 + 0i
    """
    xr, xi = 0.0, 0.0
    yr, yi = ma01ad(xr, xi)

    assert yr == pytest.approx(0.0, abs=1e-15)
    assert yi == pytest.approx(0.0, abs=1e-15)


def test_ma01ad_unity():
    """
    Test square root of 1.
    sqrt(1 + 0i) = 1 + 0i
    """
    xr, xi = 1.0, 0.0
    yr, yi = ma01ad(xr, xi)

    assert yr == pytest.approx(1.0, rel=1e-14)
    assert yi == pytest.approx(0.0, abs=1e-15)


def test_ma01ad_large_numbers():
    """
    Test with large numbers to verify overflow prevention.
    """
    xr, xi = 1e100, 1e100
    yr, yi = ma01ad(xr, xi)

    # Verify squaring property still holds
    result_real = yr * yr - yi * yi
    result_imag = 2 * yr * yi

    np.testing.assert_allclose(result_real, xr, rtol=1e-12)
    np.testing.assert_allclose(result_imag, xi, rtol=1e-12)
