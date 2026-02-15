"""
Tests for SB08MD - Spectral factorization of polynomials (continuous-time).

Computes a real polynomial E(s) such that:
  (a) E(-s) * E(s) = A(-s) * A(s)
  (b) E(s) is stable (all zeros have non-positive real parts)

Tests derived from SLICOT HTML documentation example.
"""

import numpy as np
import pytest

from ctrlsys import sb08md


class TestSB08MDBasic:
    """Basic functionality tests from HTML doc example."""

    def test_html_doc_example_acona_a(self):
        """
        Test from HTML doc: DA=3, ACONA='A', A(s) = 8 - 6*s - 3*s^2 + s^3

        Expected:
        - B(s) coefficients: [64, -84, 21, -1] (in powers of s**2)
        - E(s) coefficients: [8, 14, 7, 1]
        - RES ~ 2.7E-15
        """
        da = 3
        a = np.array([8.0, -6.0, -3.0, 1.0], dtype=float)

        e, b, res, info = sb08md('A', da, a)

        assert info == 0

        b_expected = np.array([64.0, -84.0, 21.0, -1.0])
        np.testing.assert_allclose(b, b_expected, rtol=1e-12)

        e_expected = np.array([8.0, 14.0, 7.0, 1.0])
        np.testing.assert_allclose(e, e_expected, rtol=1e-3, atol=1e-4)

        assert res < 1e-10

    def test_acona_b_direct_b_input(self):
        """
        Test ACONA='B' mode: supply B(s) coefficients directly.

        Use B(s) = A(-s)*A(s) from above example: [64, -84, 21, -1]
        Should get same E(s) = [8, 14, 7, 1]
        """
        da = 3
        b_input = np.array([64.0, -84.0, 21.0, -1.0], dtype=float)

        e, b_out, res, info = sb08md('B', da, b_input)

        assert info == 0

        np.testing.assert_allclose(b_out, b_input, rtol=1e-12)

        e_expected = np.array([8.0, 14.0, 7.0, 1.0])
        np.testing.assert_allclose(e, e_expected, rtol=1e-3, atol=1e-4)

        assert res < 1e-10


class TestSB08MDMathematicalProperties:
    """Mathematical property validation tests."""

    def test_spectral_factorization_property(self):
        """
        Validate E(-s)*E(s) = A(-s)*A(s) property.

        For polynomial E(s) = e0 + e1*s + e2*s^2 + ...
        E(-s)*E(s) has only even powers of s (B(s) form).

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)

        da = 2
        a = np.array([2.0, 1.0, 0.5], dtype=float)

        e, b, res, info = sb08md('A', da, a)
        assert info == 0

        from ctrlsys import sb08my
        epsb = np.finfo(float).eps
        b_from_e, _ = sb08my(e, epsb)
        np.testing.assert_allclose(b_from_e, b, rtol=1e-10)

    def test_stability_of_e_polynomial(self):
        """
        Validate E(s) is stable: all roots have Re <= 0.

        Use a known stable polynomial and verify output roots are stable.
        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)

        da = 3
        a = np.array([8.0, -6.0, -3.0, 1.0], dtype=float)

        e, b, res, info = sb08md('A', da, a)
        assert info == 0

        roots_e = np.roots(e[::-1])
        for root in roots_e:
            assert root.real <= 1e-10, f"Unstable root: {root}"

    def test_e_evaluated_at_one_decreasing(self):
        """
        Property: E(s) evaluated at s=1 should be positive for stable polynomial.

        This validates the spectral factor normalization.
        """
        da = 2
        a = np.array([1.0, 2.0, 1.0], dtype=float)

        e, b, res, info = sb08md('A', da, a)
        assert info == 0

        e_at_1 = np.polyval(e[::-1], 1.0)
        assert e_at_1 > 0, f"E(1) = {e_at_1} should be positive"


class TestSB08MDEdgeCases:
    """Edge case tests."""

    def test_degree_zero(self):
        """Test with DA=0 (constant polynomial)."""
        da = 0
        a = np.array([4.0], dtype=float)

        e, b, res, info = sb08md('A', da, a)

        assert info == 0
        np.testing.assert_allclose(b, [16.0], rtol=1e-14)
        np.testing.assert_allclose(e, [4.0], rtol=1e-14)

    def test_degree_one(self):
        """Test with DA=1 (linear polynomial)."""
        da = 1
        a = np.array([2.0, 1.0], dtype=float)

        e, b, res, info = sb08md('A', da, a)

        assert info == 0
        b_expected = np.array([4.0, -1.0])
        np.testing.assert_allclose(b, b_expected, rtol=1e-12)


class TestSB08MDErrorHandling:
    """Error handling tests."""

    def test_info_1_all_zeros(self):
        """Test INFO=1: all coefficients are zero."""
        da = 2
        a = np.array([0.0, 0.0, 0.0], dtype=float)

        e, b, res, info = sb08md('A', da, a)

        assert info == 1

    def test_info_2_invalid_b_coefficients(self):
        """
        Test INFO=2: B(s) coefficients not valid for spectral factorization.

        B(1) < 0 or (-1)^DA * B(DA+1) < 0 means no real solution.
        """
        da = 2
        b_invalid = np.array([-1.0, 2.0, 1.0], dtype=float)

        e, b, res, info = sb08md('B', da, b_invalid)

        assert info == 2

    def test_invalid_acona(self):
        """Test with invalid ACONA parameter."""
        da = 2
        a = np.array([1.0, 2.0, 1.0], dtype=float)

        with pytest.raises(ValueError):
            sb08md('X', da, a)

    def test_negative_da(self):
        """Test with negative DA."""
        a = np.array([1.0], dtype=float)

        with pytest.raises(ValueError):
            sb08md('A', -1, a)
