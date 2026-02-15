"""
Tests for MB01UZ - Complex triangular matrix product.

Computes T := alpha*op(T)*A or T := alpha*A*op(T) where
T is triangular (upper or lower) and A is M-by-N.
op(T) = T, T', or conj(T').
Result overwrites T.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ctrlsys import mb01uz


class TestMB01UZBasic:
    """Basic functionality tests for MB01UZ."""

    def test_left_upper_notrans(self):
        """
        Test T := alpha * T * A with upper triangular T.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        m, n = 3, 4
        alpha = 1.5 + 0.5j

        t_orig = (np.random.randn(m, m) + 1j * np.random.randn(m, m)).astype(
            complex, order='F'
        )
        t = np.triu(t_orig).copy(order='F')

        a = (np.random.randn(m, n) + 1j * np.random.randn(m, n)).astype(
            complex, order='F'
        )

        t_expected = alpha * np.triu(t_orig) @ a

        t_input = np.zeros((m, n), dtype=complex, order='F')
        t_input[:m, :m] = t

        t_result, info = mb01uz('L', 'U', 'N', m, n, alpha, t_input, a)

        assert info == 0
        assert_allclose(t_result[:m, :n], t_expected, rtol=1e-14)

    def test_left_lower_notrans(self):
        """
        Test T := alpha * T * A with lower triangular T.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        m, n = 3, 4
        alpha = 2.0 - 1.0j

        t_orig = (np.random.randn(m, m) + 1j * np.random.randn(m, m)).astype(
            complex, order='F'
        )
        t = np.tril(t_orig).copy(order='F')

        a = (np.random.randn(m, n) + 1j * np.random.randn(m, n)).astype(
            complex, order='F'
        )

        t_expected = alpha * np.tril(t_orig) @ a

        t_input = np.zeros((m, n), dtype=complex, order='F')
        t_input[:m, :m] = t

        t_result, info = mb01uz('L', 'L', 'N', m, n, alpha, t_input, a)

        assert info == 0
        assert_allclose(t_result[:m, :n], t_expected, rtol=1e-14)

    def test_right_upper_notrans(self):
        """
        Test T := alpha * A * T with upper triangular T on right.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        m, n = 3, 4
        alpha = 1.0 + 2.0j

        t_orig = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(
            complex, order='F'
        )
        t = np.triu(t_orig).copy(order='F')

        a = (np.random.randn(m, n) + 1j * np.random.randn(m, n)).astype(
            complex, order='F'
        )

        t_expected = alpha * a @ np.triu(t_orig)

        ldt = max(m, n)
        t_input = np.zeros((ldt, n), dtype=complex, order='F')
        t_input[:n, :n] = t

        t_result, info = mb01uz('R', 'U', 'N', m, n, alpha, t_input, a)

        assert info == 0
        assert_allclose(t_result[:m, :n], t_expected, rtol=1e-14)

    def test_left_upper_transpose(self):
        """
        Test T := alpha * T' * A with upper triangular T, transpose.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        m, n = 3, 3
        alpha = 0.5 + 0.5j

        t_orig = (np.random.randn(m, m) + 1j * np.random.randn(m, m)).astype(
            complex, order='F'
        )
        t = np.triu(t_orig).copy(order='F')

        a = (np.random.randn(m, n) + 1j * np.random.randn(m, n)).astype(
            complex, order='F'
        )

        t_expected = alpha * np.triu(t_orig).T @ a

        t_input = t.copy(order='F')

        t_result, info = mb01uz('L', 'U', 'T', m, n, alpha, t_input, a)

        assert info == 0
        assert_allclose(t_result[:m, :n], t_expected, rtol=1e-14)

    def test_left_upper_conjtrans(self):
        """
        Test T := alpha * conj(T') * A with upper triangular T, conjugate transpose.

        Random seed: 999 (for reproducibility)
        """
        np.random.seed(999)
        m, n = 3, 4
        alpha = 1.0 - 0.5j

        t_orig = (np.random.randn(m, m) + 1j * np.random.randn(m, m)).astype(
            complex, order='F'
        )
        t = np.triu(t_orig).copy(order='F')

        a = (np.random.randn(m, n) + 1j * np.random.randn(m, n)).astype(
            complex, order='F'
        )

        t_expected = alpha * np.triu(t_orig).conj().T @ a

        t_input = np.zeros((m, n), dtype=complex, order='F')
        t_input[:m, :m] = t

        t_result, info = mb01uz('L', 'U', 'C', m, n, alpha, t_input, a)

        assert info == 0
        assert_allclose(t_result[:m, :n], t_expected, rtol=1e-14)


class TestMB01UZZeroAlpha:
    """Tests for alpha=0 edge case."""

    def test_alpha_zero_clears_output(self):
        """When alpha=0, result should be zero regardless of input."""
        np.random.seed(111)
        m, n = 3, 4

        t = (np.random.randn(m, n) + 1j * np.random.randn(m, n)).astype(
            complex, order='F'
        )
        a = (np.random.randn(m, n) + 1j * np.random.randn(m, n)).astype(
            complex, order='F'
        )

        t_result, info = mb01uz('L', 'U', 'N', m, n, 0.0 + 0.0j, t, a)

        assert info == 0
        assert_allclose(t_result[:m, :n], np.zeros((m, n), dtype=complex), rtol=1e-14)


class TestMB01UZEdgeCases:
    """Edge cases and boundary conditions."""

    def test_m_zero(self):
        """Test with m=0."""
        t = np.zeros((1, 1), dtype=complex, order='F')
        a = np.zeros((0, 3), dtype=complex, order='F')

        t_result, info = mb01uz('L', 'U', 'N', 0, 3, 1.0 + 0.0j, t, a)

        assert info == 0

    def test_n_zero(self):
        """Test with n=0."""
        t = np.zeros((3, 1), dtype=complex, order='F')
        a = np.zeros((3, 0), dtype=complex, order='F')

        t_result, info = mb01uz('L', 'U', 'N', 3, 0, 1.0 + 0.0j, t, a)

        assert info == 0

    def test_1x1_matrix(self):
        """Test with 1x1 matrices."""
        t = np.array([[2.0 + 1.0j]], dtype=complex, order='F')
        a = np.array([[3.0 - 0.5j]], dtype=complex, order='F')
        alpha = 2.0 + 0.0j

        t_expected = alpha * t @ a

        t_result, info = mb01uz('L', 'U', 'N', 1, 1, alpha, t.copy(order='F'), a)

        assert info == 0
        assert_allclose(t_result, t_expected, rtol=1e-14)


class TestMB01UZMathematicalProperties:
    """Tests verifying mathematical properties."""

    def test_identity_alpha_one(self):
        """
        Test that alpha=1 gives T*A when T is upper triangular.

        Validates: Result = T @ A (for SIDE='L', TRANS='N')
        Random seed: 222 (for reproducibility)
        """
        np.random.seed(222)
        m, n = 3, 4
        alpha = 1.0 + 0.0j

        t_orig = (np.random.randn(m, m) + 1j * np.random.randn(m, m)).astype(
            complex, order='F'
        )
        t = np.triu(t_orig).copy(order='F')

        a = (np.random.randn(m, n) + 1j * np.random.randn(m, n)).astype(
            complex, order='F'
        )

        t_expected = np.triu(t_orig) @ a

        t_input = np.zeros((m, n), dtype=complex, order='F')
        t_input[:m, :m] = t

        t_result, info = mb01uz('L', 'U', 'N', m, n, alpha, t_input, a)

        assert info == 0
        assert_allclose(t_result[:m, :n], t_expected, rtol=1e-14)

    def test_scaling_property(self):
        """
        Test that alpha scales the result linearly.

        Validates: alpha1 * (T @ A) = result1, alpha2 * (T @ A) = result2
        and result2 / result1 = alpha2 / alpha1
        Random seed: 333 (for reproducibility)
        """
        np.random.seed(333)
        m, n = 3, 3
        alpha1 = 1.0 + 0.0j
        alpha2 = 2.5 + 1.5j

        t_orig = (np.random.randn(m, m) + 1j * np.random.randn(m, m)).astype(
            complex, order='F'
        )
        t_tri = np.triu(t_orig)

        a = (np.random.randn(m, n) + 1j * np.random.randn(m, n)).astype(
            complex, order='F'
        )

        t1 = t_tri.copy(order='F')
        t2 = t_tri.copy(order='F')

        r1, _ = mb01uz('L', 'U', 'N', m, n, alpha1, t1, a)
        r2, _ = mb01uz('L', 'U', 'N', m, n, alpha2, t2, a)

        expected_r2 = r1 * (alpha2 / alpha1)
        assert_allclose(r2, expected_r2, rtol=1e-14)

    def test_transpose_consistency(self):
        """
        Test consistency between transpose and conjugate transpose.

        For real-valued matrices, T' and conj(T') should be the same.
        Random seed: 444 (for reproducibility)
        """
        np.random.seed(444)
        m, n = 3, 3
        alpha = 1.0 + 0.0j

        t_real = np.random.randn(m, m).astype(complex, order='F')
        t_tri = np.triu(t_real)

        a_real = np.random.randn(m, n).astype(complex, order='F')

        t1 = t_tri.copy(order='F')
        t2 = t_tri.copy(order='F')

        r1, info1 = mb01uz('L', 'U', 'T', m, n, alpha, t1, a_real)
        r2, info2 = mb01uz('L', 'U', 'C', m, n, alpha, t2, a_real)

        assert info1 == 0
        assert info2 == 0
        assert_allclose(r1, r2, rtol=1e-14)


class TestMB01UZErrorHandling:
    """Tests for error handling."""

    def test_invalid_side(self):
        """Test error for invalid SIDE parameter."""
        t = np.zeros((3, 3), dtype=complex, order='F')
        a = np.zeros((3, 3), dtype=complex, order='F')

        _, info = mb01uz('X', 'U', 'N', 3, 3, 1.0 + 0.0j, t, a)

        assert info == -1

    def test_invalid_uplo(self):
        """Test error for invalid UPLO parameter."""
        t = np.zeros((3, 3), dtype=complex, order='F')
        a = np.zeros((3, 3), dtype=complex, order='F')

        _, info = mb01uz('L', 'X', 'N', 3, 3, 1.0 + 0.0j, t, a)

        assert info == -2

    def test_invalid_trans(self):
        """Test error for invalid TRANS parameter."""
        t = np.zeros((3, 3), dtype=complex, order='F')
        a = np.zeros((3, 3), dtype=complex, order='F')

        _, info = mb01uz('L', 'U', 'X', 3, 3, 1.0 + 0.0j, t, a)

        assert info == -3

    def test_negative_m(self):
        """Test error for negative M."""
        t = np.zeros((3, 3), dtype=complex, order='F')
        a = np.zeros((3, 3), dtype=complex, order='F')

        _, info = mb01uz('L', 'U', 'N', -1, 3, 1.0 + 0.0j, t, a)

        assert info == -4

    def test_negative_n(self):
        """Test error for negative N."""
        t = np.zeros((3, 3), dtype=complex, order='F')
        a = np.zeros((3, 3), dtype=complex, order='F')

        _, info = mb01uz('L', 'U', 'N', 3, -1, 1.0 + 0.0j, t, a)

        assert info == -5
