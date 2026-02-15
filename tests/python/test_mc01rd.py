"""Tests for mc01rd - polynomial P(x) = P1(x) * P2(x) + alpha * P3(x)."""

import numpy as np
import pytest
from ctrlsys import mc01rd


class TestMC01RDBasic:
    """Basic functionality tests from HTML documentation."""

    def test_html_example(self):
        """Test case from SLICOT HTML documentation.

        P1(x) = 1.0 + 2.5*x (degree 1)
        P2(x) = 1.0 + 0.1*x - 0.4*x^2 (degree 2)
        P3(x) = 1.15 + 1.5*x (degree 1)
        ALPHA = -2.2

        Expected: P(x) = P1*P2 + alpha*P3
        P1*P2 = (1 + 2.5x)(1 + 0.1x - 0.4x^2)
              = 1 + 0.1x - 0.4x^2 + 2.5x + 0.25x^2 - x^3
              = 1 + 2.6x - 0.15x^2 - x^3
        alpha*P3 = -2.2*(1.15 + 1.5x) = -2.53 - 3.3x
        P = 1 + 2.6x - 0.15x^2 - x^3 - 2.53 - 3.3x
          = -1.53 - 0.7x - 0.15x^2 - x^3

        Result degree = 3
        Result coefficients = [-1.53, -0.70, -0.15, -1.0]
        """
        p1 = np.array([1.0, 2.5], dtype=float)
        p2 = np.array([1.0, 0.1, -0.4], dtype=float)
        p3 = np.array([1.15, 1.5, 0.0, 0.0], dtype=float)
        alpha = -2.2

        p_out, dp_out, info = mc01rd(p1, p2, p3, alpha)

        assert info == 0
        assert dp_out == 3
        expected = np.array([-1.53, -0.70, -0.15, -1.0])
        np.testing.assert_allclose(p_out[:dp_out + 1], expected, rtol=1e-10)


class TestMC01RDPolynomialMultiplication:
    """Tests for polynomial multiplication (alpha=0 or P3=0)."""

    def test_simple_multiplication(self):
        """Test P(x) = P1(x) * P2(x) when alpha=0.

        P1(x) = 1 + x
        P2(x) = 1 + x
        Expected: P(x) = 1 + 2x + x^2
        """
        p1 = np.array([1.0, 1.0], dtype=float)
        p2 = np.array([1.0, 1.0], dtype=float)
        p3 = np.array([0.0, 0.0, 0.0], dtype=float)
        alpha = 0.0

        p_out, dp_out, info = mc01rd(p1, p2, p3, alpha, dp3=0)

        assert info == 0
        assert dp_out == 2
        expected = np.array([1.0, 2.0, 1.0])
        np.testing.assert_allclose(p_out[:dp_out + 1], expected, rtol=1e-14)

    def test_multiply_by_constant(self):
        """Test P(x) = c * P2(x) where P1 is a constant.

        P1(x) = 3
        P2(x) = 1 + 2x + 3x^2
        Expected: P(x) = 3 + 6x + 9x^2
        """
        p1 = np.array([3.0], dtype=float)
        p2 = np.array([1.0, 2.0, 3.0], dtype=float)
        p3 = np.array([0.0, 0.0, 0.0], dtype=float)
        alpha = 0.0

        p_out, dp_out, info = mc01rd(p1, p2, p3, alpha, dp1=0, dp3=0)

        assert info == 0
        assert dp_out == 2
        expected = np.array([3.0, 6.0, 9.0])
        np.testing.assert_allclose(p_out[:dp_out + 1], expected, rtol=1e-14)

    def test_multiply_different_degrees(self):
        """Test multiplication of polynomials with different degrees.

        P1(x) = 1 + 2x + 3x^2 (degree 2)
        P2(x) = 2 + x (degree 1)
        Expected: P(x) = 2 + 5x + 8x^2 + 3x^3
        """
        p1 = np.array([1.0, 2.0, 3.0], dtype=float)
        p2 = np.array([2.0, 1.0], dtype=float)
        p3 = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
        alpha = 0.0

        p_out, dp_out, info = mc01rd(p1, p2, p3, alpha, dp3=0)

        assert info == 0
        assert dp_out == 3
        expected = np.array([2.0, 5.0, 8.0, 3.0])
        np.testing.assert_allclose(p_out[:dp_out + 1], expected, rtol=1e-14)


class TestMC01RDAddition:
    """Tests for polynomial addition (P1 or P2 is zero)."""

    def test_alpha_times_p3_only(self):
        """Test P(x) = alpha * P3(x) when P1 is zero polynomial.

        P1(x) = 0 (degree -1)
        P2(x) = 1 + x (any polynomial)
        P3(x) = 1 + 2x + 3x^2
        alpha = 2.0
        Expected: P(x) = 2 + 4x + 6x^2
        """
        p1 = np.array([0.0], dtype=float)
        p2 = np.array([1.0, 1.0], dtype=float)
        p3 = np.array([1.0, 2.0, 3.0], dtype=float)
        alpha = 2.0

        p_out, dp_out, info = mc01rd(p1, p2, p3, alpha, dp1=-1)

        assert info == 0
        assert dp_out == 2
        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_allclose(p_out[:dp_out + 1], expected, rtol=1e-14)


class TestMC01RDCombined:
    """Tests combining multiplication and addition."""

    def test_product_plus_scaled_p3(self):
        """Test P(x) = P1*P2 + alpha*P3 with all non-zero.

        P1(x) = 1 + x
        P2(x) = 1 - x
        P1*P2 = 1 - x^2
        P3(x) = x + x^2
        alpha = 2.0
        Expected: P(x) = 1 + 2x + x^2
        """
        p1 = np.array([1.0, 1.0], dtype=float)
        p2 = np.array([1.0, -1.0], dtype=float)
        p3 = np.array([0.0, 1.0, 1.0], dtype=float)
        alpha = 2.0

        p_out, dp_out, info = mc01rd(p1, p2, p3, alpha)

        assert info == 0
        assert dp_out == 2
        expected = np.array([1.0, 2.0, 1.0])
        np.testing.assert_allclose(p_out[:dp_out + 1], expected, rtol=1e-14)

    def test_cancellation_to_lower_degree(self):
        """Test where leading coefficients cancel.

        P1(x) = x^2
        P2(x) = 1
        P1*P2 = x^2
        P3(x) = x^2
        alpha = -1.0
        Expected: P(x) = 0 (degree -1)
        """
        p1 = np.array([0.0, 0.0, 1.0], dtype=float)
        p2 = np.array([1.0], dtype=float)
        p3 = np.array([0.0, 0.0, 1.0], dtype=float)
        alpha = -1.0

        p_out, dp_out, info = mc01rd(p1, p2, p3, alpha)

        assert info == 0
        assert dp_out == -1


class TestMC01RDZeroPolynomials:
    """Tests involving zero polynomials."""

    def test_p1_zero(self):
        """Test with P1(x) = 0 (degree -1)."""
        p1 = np.array([0.0], dtype=float)
        p2 = np.array([1.0, 2.0], dtype=float)
        p3 = np.array([3.0, 4.0], dtype=float)
        alpha = 1.0

        p_out, dp_out, info = mc01rd(p1, p2, p3, alpha, dp1=-1)

        assert info == 0
        assert dp_out == 1
        expected = np.array([3.0, 4.0])
        np.testing.assert_allclose(p_out[:dp_out + 1], expected, rtol=1e-14)

    def test_p2_zero(self):
        """Test with P2(x) = 0 (degree -1)."""
        p1 = np.array([1.0, 2.0], dtype=float)
        p2 = np.array([0.0], dtype=float)
        p3 = np.array([5.0, 6.0], dtype=float)
        alpha = 0.5

        p_out, dp_out, info = mc01rd(p1, p2, p3, alpha, dp2=-1)

        assert info == 0
        assert dp_out == 1
        expected = np.array([2.5, 3.0])
        np.testing.assert_allclose(p_out[:dp_out + 1], expected, rtol=1e-14)

    def test_p3_zero(self):
        """Test with P3(x) = 0 (degree -1)."""
        p1 = np.array([1.0, 1.0], dtype=float)
        p2 = np.array([1.0, 1.0], dtype=float)
        p3 = np.array([0.0, 0.0, 0.0], dtype=float)
        alpha = 5.0

        p_out, dp_out, info = mc01rd(p1, p2, p3, alpha, dp3=-1)

        assert info == 0
        assert dp_out == 2
        expected = np.array([1.0, 2.0, 1.0])
        np.testing.assert_allclose(p_out[:dp_out + 1], expected, rtol=1e-14)

    def test_all_zero(self):
        """Test with all zero polynomials."""
        p1 = np.array([0.0], dtype=float)
        p2 = np.array([0.0], dtype=float)
        p3 = np.array([0.0], dtype=float)
        alpha = 1.0

        p_out, dp_out, info = mc01rd(p1, p2, p3, alpha, dp1=-1, dp2=-1, dp3=-1)

        assert info == 0
        assert dp_out == -1


class TestMC01RDMathematicalProperties:
    """Tests verifying mathematical properties."""

    def test_commutativity_of_multiplication(self):
        """Verify P1*P2 = P2*P1 (commutative property).

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        p1 = np.random.randn(3)
        p2 = np.random.randn(4)
        p3_zeros = np.zeros(6)
        alpha = 0.0

        result1, dp1_out, info1 = mc01rd(p1, p2, p3_zeros.copy(), alpha)
        result2, dp2_out, info2 = mc01rd(p2, p1, p3_zeros.copy(), alpha)

        assert info1 == 0 and info2 == 0
        assert dp1_out == dp2_out
        np.testing.assert_allclose(
            result1[:dp1_out + 1], result2[:dp2_out + 1], rtol=1e-14
        )

    def test_associativity_with_alpha(self):
        """Verify (P1*P2 + a*P3) computed correctly vs manual calculation.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        p1 = np.array([1.0, 2.0], dtype=float)
        p2 = np.array([3.0, 4.0], dtype=float)
        p3 = np.array([0.5, 0.5, 0.0], dtype=float)
        alpha = 2.0

        result, dp_out, info = mc01rd(p1, p2, p3.copy(), alpha)
        assert info == 0

        manual_product = np.array([3.0, 10.0, 8.0])
        manual_alpha_p3 = alpha * np.array([0.5, 0.5])
        manual_result = manual_product.copy()
        manual_result[:2] += manual_alpha_p3

        np.testing.assert_allclose(result[:dp_out + 1], manual_result, rtol=1e-14)

    def test_distributive_property(self):
        """Verify P1*(P2+P3) = P1*P2 + P1*P3 using mc01rd.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        p1 = np.array([1.0, 2.0], dtype=float)
        p2 = np.array([1.0, 1.0], dtype=float)
        p3 = np.array([2.0, 3.0], dtype=float)

        p2_plus_p3 = p2 + p3
        zeros_4 = np.zeros(4)

        lhs, dp_lhs, info1 = mc01rd(p1, p2_plus_p3, zeros_4.copy(), 0.0)
        assert info1 == 0

        p1_times_p2, dp_12, info2 = mc01rd(p1, p2, zeros_4.copy(), 0.0)
        assert info2 == 0

        rhs, dp_rhs, info3 = mc01rd(p1, p3, p1_times_p2.copy(), 1.0)
        assert info3 == 0

        assert dp_lhs == dp_rhs
        np.testing.assert_allclose(lhs[:dp_lhs + 1], rhs[:dp_rhs + 1], rtol=1e-14)


class TestMC01RDEdgeCases:
    """Edge case tests."""

    def test_degree_zero_polynomials(self):
        """Test with degree 0 polynomials (constants).

        P1(x) = 2, P2(x) = 3, P3(x) = 4, alpha = 0.5
        Expected: P(x) = 2*3 + 0.5*4 = 8
        """
        p1 = np.array([2.0], dtype=float)
        p2 = np.array([3.0], dtype=float)
        p3 = np.array([4.0], dtype=float)
        alpha = 0.5

        p_out, dp_out, info = mc01rd(p1, p2, p3, alpha, dp1=0, dp2=0, dp3=0)

        assert info == 0
        assert dp_out == 0
        np.testing.assert_allclose(p_out[0], 8.0, rtol=1e-14)

    def test_large_degree_difference(self):
        """Test with significantly different polynomial degrees.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        p1 = np.array([1.0], dtype=float)
        p2 = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
        p3 = np.zeros(5)
        alpha = 0.0

        p_out, dp_out, info = mc01rd(p1, p2, p3, alpha, dp1=0)

        assert info == 0
        assert dp_out == 4
        np.testing.assert_allclose(p_out[:5], p2, rtol=1e-14)

    def test_leading_zeros_trimmed(self):
        """Test that leading zeros in result are trimmed.

        P1(x) = x, P2(x) = 1, P3(x) = x
        P1*P2 = x
        alpha*P3 = -1*x with alpha=-1
        Result: x - x = 0 (degree -1, zero polynomial)
        """
        p1 = np.array([0.0, 1.0], dtype=float)
        p2 = np.array([1.0], dtype=float)
        p3 = np.array([0.0, 1.0], dtype=float)
        alpha = -1.0

        p_out, dp_out, info = mc01rd(p1, p2, p3, alpha, dp2=0)

        assert info == 0
        assert dp_out == -1


class TestMC01RDErrorHandling:
    """Error handling tests."""

    def test_invalid_dp1(self):
        """Test error when dp1 < -1."""
        p1 = np.array([1.0], dtype=float)
        p2 = np.array([1.0], dtype=float)
        p3 = np.array([1.0], dtype=float)
        alpha = 1.0

        p_out, dp_out, info = mc01rd(p1, p2, p3, alpha, dp1=-2)

        assert info == -1

    def test_invalid_dp2(self):
        """Test error when dp2 < -1."""
        p1 = np.array([1.0], dtype=float)
        p2 = np.array([1.0], dtype=float)
        p3 = np.array([1.0], dtype=float)
        alpha = 1.0

        p_out, dp_out, info = mc01rd(p1, p2, p3, alpha, dp2=-2)

        assert info == -2

    def test_invalid_dp3(self):
        """Test error when dp3 < -1."""
        p1 = np.array([1.0], dtype=float)
        p2 = np.array([1.0], dtype=float)
        p3 = np.array([1.0], dtype=float)
        alpha = 1.0

        p_out, dp_out, info = mc01rd(p1, p2, p3, alpha, dp3=-2)

        assert info == -3
