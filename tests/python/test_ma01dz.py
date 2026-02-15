"""
Tests for MA01DZ - Approximate symmetric chordal metric for two complex numbers.

MA01DZ computes an approximate symmetric chordal metric for two complex
numbers A1 and A2 represented as rational numbers (numerator/denominator).

The chordal metric formula:
    D = MIN(|A1 - A2|, |1/A1 - 1/A2|)

Return values:
    D1 = numerator of chordal metric (D1 >= 0)
    D2 = denominator (0 or 1)
    iwarn = 0 (success) or 1 (NaN input)

Special cases:
    - B=0 with nonzero numerator means infinity
    - AR=AI=B=0 means NaN (not a number)
    - D2=0 when result is infinity
    - D1=D2=0 means undefined (NaN inputs)

Test data sources:
- Mathematical properties of chordal metric
- Known special cases (infinity, zero, NaN)
"""

import numpy as np
import pytest

from ctrlsys import ma01dz


class TestMA01DZBasic:
    """Basic functionality tests."""

    def test_identical_numbers(self):
        """Chordal metric of identical numbers is 0."""
        eps = np.finfo(float).eps
        safemn = np.finfo(float).tiny

        d1, d2, iwarn = ma01dz(1.0, 2.0, 3.0, 1.0, 2.0, 3.0, eps, safemn)

        assert iwarn == 0
        assert d2 == 1.0
        assert d1 == 0.0

    def test_simple_real_numbers(self):
        """Chordal metric of two real numbers: A1=2, A2=3."""
        eps = np.finfo(float).eps
        safemn = np.finfo(float).tiny

        d1, d2, iwarn = ma01dz(2.0, 0.0, 1.0, 3.0, 0.0, 1.0, eps, safemn)

        assert iwarn == 0
        assert d2 == 1.0
        # |A1 - A2| = |2 - 3| = 1
        # |1/A1 - 1/A2| = |0.5 - 0.333...| = 0.166...
        # D = min(1, 0.166...) = 0.166...
        expected = abs(1.0/2.0 - 1.0/3.0)
        np.testing.assert_allclose(d1, expected, rtol=1e-14)

    def test_simple_complex_numbers(self):
        """Chordal metric of two complex numbers."""
        eps = np.finfo(float).eps
        safemn = np.finfo(float).tiny

        # A1 = (1 + 2j)/1 = 1+2j
        # A2 = (3 + 4j)/1 = 3+4j
        d1, d2, iwarn = ma01dz(1.0, 2.0, 1.0, 3.0, 4.0, 1.0, eps, safemn)

        assert iwarn == 0
        assert d2 == 1.0

        a1 = complex(1.0, 2.0)
        a2 = complex(3.0, 4.0)
        diff = abs(a1 - a2)
        inv_diff = abs(1.0/a1 - 1.0/a2)
        expected = min(diff, inv_diff)

        np.testing.assert_allclose(d1, expected, rtol=1e-14)


class TestMA01DZSpecialCases:
    """Tests for special cases (infinity, zero, NaN)."""

    def test_both_infinite(self):
        """Both numbers infinite: D = 0."""
        eps = np.finfo(float).eps
        safemn = np.finfo(float).tiny

        # A1 = 1/0 = inf, A2 = 2/0 = inf
        d1, d2, iwarn = ma01dz(1.0, 0.0, 0.0, 2.0, 0.0, 0.0, eps, safemn)

        assert iwarn == 0
        assert d2 == 1.0
        assert d1 == 0.0

    def test_both_zero(self):
        """Both numbers zero: D = 0."""
        eps = np.finfo(float).eps
        safemn = np.finfo(float).tiny

        # A1 = 0/1 = 0, A2 = 0/1 = 0
        d1, d2, iwarn = ma01dz(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, eps, safemn)

        assert iwarn == 0
        assert d2 == 1.0
        assert d1 == 0.0

    def test_one_infinite_one_finite(self):
        """One infinite, one finite nonzero: D = 1/|A_finite|."""
        eps = np.finfo(float).eps
        safemn = np.finfo(float).tiny

        # A1 = 1/0 = inf, A2 = (3+4j)/1 = 3+4j (|A2| = 5)
        d1, d2, iwarn = ma01dz(1.0, 0.0, 0.0, 3.0, 4.0, 1.0, eps, safemn)

        assert iwarn == 0
        assert d2 == 1.0
        expected = 1.0 / 5.0
        np.testing.assert_allclose(d1, expected, rtol=1e-14)

    def test_one_zero_one_finite(self):
        """One zero, one finite nonzero: D = |A_nonzero|."""
        eps = np.finfo(float).eps
        safemn = np.finfo(float).tiny

        # A1 = 0/1 = 0, A2 = (3+4j)/1 = 3+4j (|A2| = 5)
        d1, d2, iwarn = ma01dz(0.0, 0.0, 1.0, 3.0, 4.0, 1.0, eps, safemn)

        assert iwarn == 0
        assert d2 == 1.0
        expected = 5.0
        np.testing.assert_allclose(d1, expected, rtol=1e-14)

    def test_nan_first_argument(self):
        """First argument is NaN (AR1=AI1=B1=0): iwarn=1."""
        eps = np.finfo(float).eps
        safemn = np.finfo(float).tiny

        d1, d2, iwarn = ma01dz(0.0, 0.0, 0.0, 1.0, 0.0, 1.0, eps, safemn)

        assert iwarn == 1
        assert d1 == 0.0
        assert d2 == 0.0

    def test_nan_second_argument(self):
        """Second argument is NaN (AR2=AI2=B2=0): iwarn=1."""
        eps = np.finfo(float).eps
        safemn = np.finfo(float).tiny

        d1, d2, iwarn = ma01dz(1.0, 0.0, 1.0, 0.0, 0.0, 0.0, eps, safemn)

        assert iwarn == 1
        assert d1 == 0.0
        assert d2 == 0.0

    def test_both_nan(self):
        """Both arguments are NaN: iwarn=1."""
        eps = np.finfo(float).eps
        safemn = np.finfo(float).tiny

        d1, d2, iwarn = ma01dz(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, eps, safemn)

        assert iwarn == 1
        assert d1 == 0.0
        assert d2 == 0.0


class TestMA01DZRationalRepresentation:
    """Tests using non-trivial rational representations."""

    def test_rational_form(self):
        """Test rational number representation: A = (AR+i*AI)/B."""
        eps = np.finfo(float).eps
        safemn = np.finfo(float).tiny

        # A1 = (6 + 0j)/2 = 3
        # A2 = (10 + 0j)/2 = 5
        d1, d2, iwarn = ma01dz(6.0, 0.0, 2.0, 10.0, 0.0, 2.0, eps, safemn)

        assert iwarn == 0
        assert d2 == 1.0
        # |A1 - A2| = |3 - 5| = 2
        # |1/A1 - 1/A2| = |1/3 - 1/5| = |2/15| = 0.133...
        expected = abs(1.0/3.0 - 1.0/5.0)
        np.testing.assert_allclose(d1, expected, rtol=1e-14)

    def test_complex_rational(self):
        """Test complex rational numbers."""
        eps = np.finfo(float).eps
        safemn = np.finfo(float).tiny

        # A1 = (2 + 4j)/2 = 1 + 2j
        # A2 = (6 + 8j)/2 = 3 + 4j
        d1, d2, iwarn = ma01dz(2.0, 4.0, 2.0, 6.0, 8.0, 2.0, eps, safemn)

        assert iwarn == 0
        assert d2 == 1.0

        a1 = complex(1.0, 2.0)
        a2 = complex(3.0, 4.0)
        diff = abs(a1 - a2)
        inv_diff = abs(1.0/a1 - 1.0/a2)
        expected = min(diff, inv_diff)

        np.testing.assert_allclose(d1, expected, rtol=1e-14)


class TestMA01DZMathProperties:
    """Mathematical property validation tests."""

    def test_symmetry(self):
        """
        Chordal metric is symmetric: D(A1, A2) = D(A2, A1).

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        eps = np.finfo(float).eps
        safemn = np.finfo(float).tiny

        for _ in range(10):
            ar1 = np.random.uniform(-10, 10)
            ai1 = np.random.uniform(-10, 10)
            b1 = np.random.uniform(0.1, 10)
            ar2 = np.random.uniform(-10, 10)
            ai2 = np.random.uniform(-10, 10)
            b2 = np.random.uniform(0.1, 10)

            d1_12, d2_12, iwarn_12 = ma01dz(ar1, ai1, b1, ar2, ai2, b2, eps, safemn)
            d1_21, d2_21, iwarn_21 = ma01dz(ar2, ai2, b2, ar1, ai1, b1, eps, safemn)

            assert iwarn_12 == iwarn_21
            assert d2_12 == d2_21
            np.testing.assert_allclose(d1_12, d1_21, rtol=1e-14)

    def test_non_negativity(self):
        """
        Chordal metric is non-negative: D >= 0.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        eps = np.finfo(float).eps
        safemn = np.finfo(float).tiny

        for _ in range(10):
            ar1 = np.random.uniform(-10, 10)
            ai1 = np.random.uniform(-10, 10)
            b1 = np.random.uniform(0.1, 10)
            ar2 = np.random.uniform(-10, 10)
            ai2 = np.random.uniform(-10, 10)
            b2 = np.random.uniform(0.1, 10)

            d1, d2, _ = ma01dz(ar1, ai1, b1, ar2, ai2, b2, eps, safemn)

            assert d1 >= 0.0
            assert d2 >= 0.0

    def test_identity_of_indiscernibles(self):
        """
        Chordal metric D(A1, A2) = 0 iff A1 = A2.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        eps = np.finfo(float).eps
        safemn = np.finfo(float).tiny

        for _ in range(10):
            ar = np.random.uniform(-10, 10)
            ai = np.random.uniform(-10, 10)
            b = np.random.uniform(0.1, 10)

            d1, d2, iwarn = ma01dz(ar, ai, b, ar, ai, b, eps, safemn)

            assert iwarn == 0
            assert d2 == 1.0
            np.testing.assert_allclose(d1, 0.0, atol=1e-15)

    def test_chordal_metric_formula(self):
        """
        Validate chordal metric formula: D = MIN(|A1-A2|, |1/A1 - 1/A2|).

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        eps = np.finfo(float).eps
        safemn = np.finfo(float).tiny

        for _ in range(10):
            ar1 = np.random.uniform(0.5, 5)
            ai1 = np.random.uniform(-2, 2)
            b1 = np.random.uniform(0.5, 2)
            ar2 = np.random.uniform(0.5, 5)
            ai2 = np.random.uniform(-2, 2)
            b2 = np.random.uniform(0.5, 2)

            d1, d2, iwarn = ma01dz(ar1, ai1, b1, ar2, ai2, b2, eps, safemn)

            assert iwarn == 0
            assert d2 == 1.0

            a1 = complex(ar1/b1, ai1/b1)
            a2 = complex(ar2/b2, ai2/b2)
            diff = abs(a1 - a2)
            inv_diff = abs(1.0/a1 - 1.0/a2)
            expected = min(diff, inv_diff)

            np.testing.assert_allclose(d1, expected, rtol=1e-13)

    def test_bounded_by_direct_difference(self):
        """
        Chordal metric is bounded: D <= |A1 - A2|.

        Random seed: 111 (for reproducibility)
        """
        np.random.seed(111)
        eps = np.finfo(float).eps
        safemn = np.finfo(float).tiny

        for _ in range(10):
            ar1 = np.random.uniform(0.5, 5)
            ai1 = np.random.uniform(-2, 2)
            b1 = np.random.uniform(0.5, 2)
            ar2 = np.random.uniform(0.5, 5)
            ai2 = np.random.uniform(-2, 2)
            b2 = np.random.uniform(0.5, 2)

            d1, d2, iwarn = ma01dz(ar1, ai1, b1, ar2, ai2, b2, eps, safemn)

            assert iwarn == 0
            assert d2 == 1.0

            a1 = complex(ar1/b1, ai1/b1)
            a2 = complex(ar2/b2, ai2/b2)
            diff = abs(a1 - a2)

            assert d1 <= diff + 1e-14 * diff


class TestMA01DZEdgeCases:
    """Edge case tests."""

    def test_very_small_denominator(self):
        """Test with very small (but nonzero) denominator approaching infinity."""
        eps = np.finfo(float).eps
        safemn = np.finfo(float).tiny

        # A1 = 1 / 1e-100 (very large)
        # A2 = 1 / 1 = 1
        d1, d2, iwarn = ma01dz(1.0, 0.0, 1e-100, 1.0, 0.0, 1.0, eps, safemn)

        assert iwarn == 0
        # Should compute |1/A1 - 1/A2| = |1e-100 - 1| ~ 1

    def test_pure_imaginary(self):
        """Test with pure imaginary numbers."""
        eps = np.finfo(float).eps
        safemn = np.finfo(float).tiny

        # A1 = 0 + 3j, A2 = 0 + 5j
        d1, d2, iwarn = ma01dz(0.0, 3.0, 1.0, 0.0, 5.0, 1.0, eps, safemn)

        assert iwarn == 0
        assert d2 == 1.0

        a1 = 3.0j
        a2 = 5.0j
        diff = abs(a1 - a2)
        inv_diff = abs(1.0/a1 - 1.0/a2)
        expected = min(diff, inv_diff)

        np.testing.assert_allclose(d1, expected, rtol=1e-14)

    def test_conjugate_pair(self):
        """Test with conjugate pair."""
        eps = np.finfo(float).eps
        safemn = np.finfo(float).tiny

        # A1 = 3 + 4j, A2 = 3 - 4j
        d1, d2, iwarn = ma01dz(3.0, 4.0, 1.0, 3.0, -4.0, 1.0, eps, safemn)

        assert iwarn == 0
        assert d2 == 1.0

        a1 = complex(3.0, 4.0)
        a2 = complex(3.0, -4.0)
        diff = abs(a1 - a2)  # = 8
        inv_diff = abs(1.0/a1 - 1.0/a2)  # = 8/25 = 0.32
        expected = min(diff, inv_diff)

        np.testing.assert_allclose(d1, expected, rtol=1e-14)
