"""
Tests for SB08ND - Spectral factorization of polynomials (discrete-time case).

Computes a real polynomial E(z) such that:
  (a) E(1/z) * E(z) = A(1/z) * A(z)
  (b) E(z) is stable (all zeros have modulus <= 1)

Tests derived from SLICOT HTML documentation example.
"""

import numpy as np
import pytest

from ctrlsys import sb08nd


class TestSB08NDBasic:
    """Basic functionality tests from HTML doc example."""

    def test_html_doc_example_acona_a(self):
        """
        Test from HTML doc: DA=2, ACONA='A', A(z) = 2 + 4.5*z + z^2

        Expected:
        - B(z) coefficients: [25.25, 13.5, 2.0]
        - E(z) coefficients: [0.5, 3.0, 4.0]
        - RES ~ 4.4E-16
        """
        da = 2
        a = np.array([2.0, 4.5, 1.0], dtype=float)

        e, b, res, info = sb08nd('A', da, a)

        assert info == 0

        b_expected = np.array([25.25, 13.5, 2.0])
        np.testing.assert_allclose(b, b_expected, rtol=1e-12)

        e_expected = np.array([0.5, 3.0, 4.0])
        np.testing.assert_allclose(e, e_expected, rtol=1e-3, atol=1e-4)

        assert res < 1e-10

    def test_acona_b_direct_b_input(self):
        """
        Test ACONA='B' mode: supply B(z) coefficients directly.

        Use B(z) = A(1/z)*A(z) from above example: [25.25, 13.5, 2.0]
        Should get same E(z) = [0.5, 3.0, 4.0]
        """
        da = 2
        b_input = np.array([25.25, 13.5, 2.0], dtype=float)

        e, b_out, res, info = sb08nd('B', da, b_input)

        assert info == 0

        np.testing.assert_allclose(b_out, b_input, rtol=1e-12)

        e_expected = np.array([0.5, 3.0, 4.0])
        np.testing.assert_allclose(e, e_expected, rtol=1e-3, atol=1e-4)

        assert res < 1e-10


class TestSB08NDMathematicalProperties:
    """Mathematical property validation tests."""

    def test_spectral_factorization_property(self):
        """
        Validate E(1/z)*E(z) = A(1/z)*A(z) property.

        Compute B from E using sb08ny and check it matches B from A.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)

        da = 2
        a = np.array([2.0, 4.5, 1.0], dtype=float)

        e, b, res, info = sb08nd('A', da, a)
        assert info == 0

        from ctrlsys import sb08ny
        b_from_e, _ = sb08ny(e)
        np.testing.assert_allclose(b_from_e, b, rtol=1e-10)

    def test_stability_of_e_polynomial(self):
        """
        Validate E(z) is stable: all roots have modulus <= 1.

        For discrete-time, stable means all zeros inside or on unit circle.
        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)

        da = 2
        a = np.array([2.0, 4.5, 1.0], dtype=float)

        e, b, res, info = sb08nd('A', da, a)
        assert info == 0

        roots_e = np.roots(e[::-1])
        for root in roots_e:
            assert np.abs(root) <= 1.0 + 1e-10, f"Unstable root: {root}, |root|={np.abs(root)}"

    def test_polynomial_reversal_relationship(self):
        """
        Verify the conjugate polynomial relationship.

        For polynomial E(z) = e0 + e1*z + e2*z^2, the conjugate is
        E(1/z) * z^da = e2 + e1*z + e0*z^2 (reversed coefficients).

        So E(1/z)*E(z) has symmetric coefficients around the middle.
        """
        da = 2
        a = np.array([2.0, 4.5, 1.0], dtype=float)

        e, b, res, info = sb08nd('A', da, a)
        assert info == 0

        e_reversed = e[::-1].copy()
        conv_full = np.convolve(e, e_reversed)

        b_full = np.zeros(2 * da + 1)
        for i in range(da + 1):
            if i == 0:
                b_full[da] = b[0]
            else:
                b_full[da + i] = b[i]
                b_full[da - i] = b[i]

        np.testing.assert_allclose(conv_full, b_full, rtol=1e-10)


class TestSB08NDEdgeCases:
    """Edge case tests."""

    def test_degree_zero(self):
        """Test with DA=0 (constant polynomial)."""
        da = 0
        a = np.array([4.0], dtype=float)

        e, b, res, info = sb08nd('A', da, a)

        assert info == 0
        np.testing.assert_allclose(b, [16.0], rtol=1e-14)
        np.testing.assert_allclose(e, [4.0], rtol=1e-14)

    def test_degree_one(self):
        """Test with DA=1 (linear polynomial).

        A(z) = 2 + z
        B(z) = A(1/z) * A(z) = b(0) + b(1)*(z + 1/z)
        b[k] = sum_{j=0}^{da-k} a[j]*a[j+k] (autocorrelation)
        b[0] = 2*2 + 1*1 = 5
        b[1] = 2*1 = 2
        """
        da = 1
        a = np.array([2.0, 1.0], dtype=float)

        e, b, res, info = sb08nd('A', da, a)

        assert info == 0

        b_expected = np.array([5.0, 2.0])
        np.testing.assert_allclose(b, b_expected, rtol=1e-12)


class TestSB08NDErrorHandling:
    """Error handling tests."""

    def test_info_2_invalid_b_coefficients(self):
        """
        Test INFO=2: B(z) coefficients not valid for spectral factorization.

        B(1) <= 0 means no real solution exists.
        """
        da = 2
        b_invalid = np.array([-1.0, 2.0, 1.0], dtype=float)

        e, b, res, info = sb08nd('B', da, b_invalid)

        assert info == 2

    def test_invalid_acona(self):
        """Test with invalid ACONA parameter."""
        da = 2
        a = np.array([1.0, 2.0, 1.0], dtype=float)

        with pytest.raises(ValueError):
            sb08nd('X', da, a)

    def test_negative_da(self):
        """Test with negative DA."""
        a = np.array([1.0], dtype=float)

        with pytest.raises(ValueError):
            sb08nd('A', -1, a)

    def test_ldwork_too_small_handled_internally(self):
        """Workspace is allocated internally so ldwork errors are hidden."""
        da = 2
        a = np.array([2.0, 4.5, 1.0], dtype=float)

        e, b, res, info = sb08nd('A', da, a)
        assert info == 0
