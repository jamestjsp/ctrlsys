"""
Tests for MA01DD - Approximate symmetric chordal metric for two complex numbers.

MA01DD computes an approximate symmetric chordal metric using:
    D = MIN(|A1 - A2|, |1/A1 - 1/A2|)

The chordal metric is finite even if both numbers are infinite,
or if one is infinite and the other is finite and nonzero.

Test data sources:
- Mathematical properties of chordal metric
- Known special cases
"""

import numpy as np
import pytest

from ctrlsys import ma01dd


def get_machine_constants():
    """Get EPS and SAFEMN machine constants."""
    eps = np.finfo(np.float64).eps
    safemn = np.finfo(np.float64).tiny
    return eps, safemn


class TestMA01DDBasic:
    """Basic functionality tests."""

    def test_identical_numbers(self):
        """
        Test that identical complex numbers have zero distance.
        D(A, A) = 0 for any A.
        """
        eps, safemn = get_machine_constants()
        ar1, ai1 = 3.0, 4.0
        ar2, ai2 = 3.0, 4.0

        d = ma01dd(ar1, ai1, ar2, ai2, eps, safemn)

        assert d == pytest.approx(0.0, abs=1e-15)

    def test_real_numbers_simple(self):
        """
        Test chordal metric for two real numbers.
        A1 = 2+0i, A2 = 5+0i
        |A1 - A2| = 3
        |1/A1 - 1/A2| = |0.5 - 0.2| = 0.3
        D = min(3, 0.3) = 0.3
        """
        eps, safemn = get_machine_constants()
        ar1, ai1 = 2.0, 0.0
        ar2, ai2 = 5.0, 0.0

        d = ma01dd(ar1, ai1, ar2, ai2, eps, safemn)

        d1 = abs(ar1 - ar2)
        d2 = abs(1.0 / ar1 - 1.0 / ar2)
        expected = min(d1, d2)

        assert d == pytest.approx(expected, rel=1e-14)

    def test_pure_imaginary_numbers(self):
        """
        Test chordal metric for pure imaginary numbers.
        A1 = 2i, A2 = 4i
        |A1 - A2| = 2
        |1/A1 - 1/A2| = |-0.5i - (-0.25i)| = 0.25
        D = min(2, 0.25) = 0.25
        """
        eps, safemn = get_machine_constants()
        ar1, ai1 = 0.0, 2.0
        ar2, ai2 = 0.0, 4.0

        d = ma01dd(ar1, ai1, ar2, ai2, eps, safemn)

        a1 = complex(ar1, ai1)
        a2 = complex(ar2, ai2)
        d1 = abs(a1 - a2)
        d2 = abs(1.0 / a1 - 1.0 / a2)
        expected = min(d1, d2)

        assert d == pytest.approx(expected, rel=1e-14)

    def test_complex_numbers_general(self):
        """
        Test chordal metric for general complex numbers.
        A1 = 1+2i, A2 = 3+4i
        """
        eps, safemn = get_machine_constants()
        ar1, ai1 = 1.0, 2.0
        ar2, ai2 = 3.0, 4.0

        d = ma01dd(ar1, ai1, ar2, ai2, eps, safemn)

        a1 = complex(ar1, ai1)
        a2 = complex(ar2, ai2)
        d1 = abs(a1 - a2)
        d2 = abs(1.0 / a1 - 1.0 / a2)
        expected = min(d1, d2)

        assert d == pytest.approx(expected, rel=1e-13)


class TestMA01DDZeros:
    """Tests involving zero values."""

    def test_both_zero(self):
        """
        Test when both numbers are zero.
        D(0, 0) = 0
        """
        eps, safemn = get_machine_constants()

        d = ma01dd(0.0, 0.0, 0.0, 0.0, eps, safemn)

        assert d == pytest.approx(0.0, abs=1e-15)

    def test_one_zero_one_nonzero(self):
        """
        Test when one number is zero.
        D(0, A) = |A| for nonzero A (since 1/0 is infinite)
        """
        eps, safemn = get_machine_constants()
        ar1, ai1 = 0.0, 0.0
        ar2, ai2 = 3.0, 4.0

        d = ma01dd(ar1, ai1, ar2, ai2, eps, safemn)

        expected = np.sqrt(ar2**2 + ai2**2)
        assert d == pytest.approx(expected, rel=1e-14)

    def test_one_nonzero_one_zero(self):
        """
        Test symmetric case: D(A, 0) = |A|
        """
        eps, safemn = get_machine_constants()
        ar1, ai1 = 3.0, 4.0
        ar2, ai2 = 0.0, 0.0

        d = ma01dd(ar1, ai1, ar2, ai2, eps, safemn)

        expected = np.sqrt(ar1**2 + ai1**2)
        assert d == pytest.approx(expected, rel=1e-14)


class TestMA01DDProperties:
    """Mathematical property tests."""

    def test_symmetry(self):
        """
        Property: D(A1, A2) = D(A2, A1) (symmetric metric)

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        eps, safemn = get_machine_constants()

        for _ in range(10):
            ar1, ai1 = np.random.randn() * 10, np.random.randn() * 10
            ar2, ai2 = np.random.randn() * 10, np.random.randn() * 10

            d1 = ma01dd(ar1, ai1, ar2, ai2, eps, safemn)
            d2 = ma01dd(ar2, ai2, ar1, ai1, eps, safemn)

            np.testing.assert_allclose(d1, d2, rtol=1e-14)

    def test_nonnegative(self):
        """
        Property: D(A1, A2) >= 0 for all inputs.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        eps, safemn = get_machine_constants()

        for _ in range(20):
            ar1, ai1 = np.random.randn() * 100, np.random.randn() * 100
            ar2, ai2 = np.random.randn() * 100, np.random.randn() * 100

            d = ma01dd(ar1, ai1, ar2, ai2, eps, safemn)

            assert d >= 0.0, f"Distance should be non-negative, got {d}"

    def test_identity_of_indiscernibles(self):
        """
        Property: D(A1, A2) = 0 implies A1 = A2

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        eps, safemn = get_machine_constants()

        for _ in range(10):
            ar1, ai1 = np.random.randn() * 10, np.random.randn() * 10

            d = ma01dd(ar1, ai1, ar1, ai1, eps, safemn)

            assert d == pytest.approx(0.0, abs=1e-15)


class TestMA01DDLargeNumbers:
    """Tests with large numbers (overflow prevention)."""

    def test_large_numbers(self):
        """
        Test that large numbers are handled without overflow.
        For very large A1 and A2, the reciprocal distance is used.
        """
        eps, safemn = get_machine_constants()
        ar1, ai1 = 1e150, 1e150
        ar2, ai2 = 2e150, 2e150

        d = ma01dd(ar1, ai1, ar2, ai2, eps, safemn)

        assert np.isfinite(d)
        assert d >= 0.0

    def test_one_large_one_small(self):
        """
        Test with one large and one small number.
        """
        eps, safemn = get_machine_constants()
        ar1, ai1 = 1e150, 0.0
        ar2, ai2 = 1.0, 0.0

        d = ma01dd(ar1, ai1, ar2, ai2, eps, safemn)

        assert np.isfinite(d)
        assert d >= 0.0

    def test_both_very_large_close(self):
        """
        Test two very large numbers that are close together.
        The reciprocal distance 1/A1 - 1/A2 should be used.
        """
        eps, safemn = get_machine_constants()
        ar1, ai1 = 1e100, 0.0
        ar2, ai2 = 1.0001e100, 0.0

        d = ma01dd(ar1, ai1, ar2, ai2, eps, safemn)

        assert np.isfinite(d)
        assert d >= 0.0


class TestMA01DDSmallNumbers:
    """Tests with small numbers (underflow prevention)."""

    def test_small_numbers(self):
        """
        Test that small numbers are handled correctly.
        """
        eps, safemn = get_machine_constants()
        ar1, ai1 = 1e-150, 1e-150
        ar2, ai2 = 2e-150, 2e-150

        d = ma01dd(ar1, ai1, ar2, ai2, eps, safemn)

        assert np.isfinite(d)
        assert d >= 0.0

    def test_one_small_one_normal(self):
        """
        Test with one small and one normal number.
        """
        eps, safemn = get_machine_constants()
        ar1, ai1 = 1e-150, 0.0
        ar2, ai2 = 1.0, 0.0

        d = ma01dd(ar1, ai1, ar2, ai2, eps, safemn)

        assert np.isfinite(d)
        assert d == pytest.approx(1.0, rel=1e-10)


class TestMA01DDUnitCircle:
    """Tests for numbers on the unit circle."""

    def test_unit_circle_points(self):
        """
        Test chordal metric for numbers on the unit circle.
        For |A1| = |A2| = 1, |1/A1 - 1/A2| = |A1 - A2| (since 1/A = conj(A) on unit circle)
        """
        eps, safemn = get_machine_constants()
        theta1, theta2 = np.pi / 4, np.pi / 3

        ar1, ai1 = np.cos(theta1), np.sin(theta1)
        ar2, ai2 = np.cos(theta2), np.sin(theta2)

        d = ma01dd(ar1, ai1, ar2, ai2, eps, safemn)

        a1 = complex(ar1, ai1)
        a2 = complex(ar2, ai2)
        expected = abs(a1 - a2)

        assert d == pytest.approx(expected, rel=1e-14)

    def test_opposite_unit_circle(self):
        """
        Test for diametrically opposite points on unit circle.
        A1 = 1, A2 = -1
        """
        eps, safemn = get_machine_constants()
        ar1, ai1 = 1.0, 0.0
        ar2, ai2 = -1.0, 0.0

        d = ma01dd(ar1, ai1, ar2, ai2, eps, safemn)

        assert d == pytest.approx(2.0, rel=1e-14)
