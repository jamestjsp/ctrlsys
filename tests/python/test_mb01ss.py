"""
Tests for MB01SS: Symmetric scaling of a symmetric matrix.

MB01SS scales a symmetric N-by-N matrix A using diagonal scaling factors D:
  - JOBS='D': A := diag(D)*A*diag(D)
  - JOBS='I': A := inv(diag(D))*A*inv(diag(D))
"""

import numpy as np
import pytest
from ctrlsys import mb01ss


class TestMB01SSBasic:
    """Basic functionality tests."""

    def test_scale_with_d_upper(self):
        """
        Test scaling with D (JOBS='D') for upper triangle.

        For a symmetric matrix A and diagonal D, the result is:
        A_out[i,j] = D[i] * A[i,j] * D[j]

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n = 3
        a_full = np.random.randn(n, n)
        a = (a_full + a_full.T) / 2
        a = np.asfortranarray(a)
        d = np.array([2.0, 3.0, 0.5])

        a_result = mb01ss('D', 'U', a.copy(order='F'), d)

        d_mat = np.diag(d)
        expected = d_mat @ a @ d_mat

        np.testing.assert_allclose(
            np.triu(a_result), np.triu(expected), rtol=1e-14
        )

    def test_scale_with_d_lower(self):
        """
        Test scaling with D (JOBS='D') for lower triangle.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n = 4
        a_full = np.random.randn(n, n)
        a = (a_full + a_full.T) / 2
        a = np.asfortranarray(a)
        d = np.array([1.5, 2.0, 0.5, 3.0])

        a_result = mb01ss('D', 'L', a.copy(order='F'), d)

        d_mat = np.diag(d)
        expected = d_mat @ a @ d_mat

        np.testing.assert_allclose(
            np.tril(a_result), np.tril(expected), rtol=1e-14
        )

    def test_scale_with_inv_d_upper(self):
        """
        Test scaling with inv(D) (JOBS='I') for upper triangle.

        For a symmetric matrix A and diagonal D, the result is:
        A_out[i,j] = (1/D[i]) * A[i,j] * (1/D[j])

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n = 3
        a_full = np.random.randn(n, n)
        a = (a_full + a_full.T) / 2
        a = np.asfortranarray(a)
        d = np.array([2.0, 4.0, 0.5])

        a_result = mb01ss('I', 'U', a.copy(order='F'), d)

        d_inv_mat = np.diag(1.0 / d)
        expected = d_inv_mat @ a @ d_inv_mat

        np.testing.assert_allclose(
            np.triu(a_result), np.triu(expected), rtol=1e-14
        )

    def test_scale_with_inv_d_lower(self):
        """
        Test scaling with inv(D) (JOBS='I') for lower triangle.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n = 4
        a_full = np.random.randn(n, n)
        a = (a_full + a_full.T) / 2
        a = np.asfortranarray(a)
        d = np.array([1.0, 2.0, 3.0, 4.0])

        a_result = mb01ss('I', 'L', a.copy(order='F'), d)

        d_inv_mat = np.diag(1.0 / d)
        expected = d_inv_mat @ a @ d_inv_mat

        np.testing.assert_allclose(
            np.tril(a_result), np.tril(expected), rtol=1e-14
        )


class TestMB01SSMathProperties:
    """Mathematical property tests."""

    def test_involution_d_then_inv_d(self):
        """
        Verify that scaling with D then inv(D) returns original matrix.

        Mathematical property: inv(diag(D)) * (diag(D)*A*diag(D)) * inv(diag(D)) = A

        Random seed: 100 (for reproducibility)
        """
        np.random.seed(100)
        n = 4
        a_full = np.random.randn(n, n)
        a = (a_full + a_full.T) / 2
        a = np.asfortranarray(a)
        d = np.array([2.0, 0.5, 3.0, 1.5])
        a_orig = a.copy(order='F')

        a_scaled = mb01ss('D', 'U', a.copy(order='F'), d)
        a_back = mb01ss('I', 'U', a_scaled.copy(order='F'), d)

        np.testing.assert_allclose(
            np.triu(a_back), np.triu(a_orig), rtol=1e-13
        )

    def test_symmetry_preservation(self):
        """
        Verify that scaling preserves symmetry of the matrix.

        If A is symmetric, diag(D)*A*diag(D) is also symmetric.

        Random seed: 200 (for reproducibility)
        """
        np.random.seed(200)
        n = 5
        a_full = np.random.randn(n, n)
        a = (a_full + a_full.T) / 2
        a = np.asfortranarray(a)
        d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        a_scaled_upper = mb01ss('D', 'U', a.copy(order='F'), d)
        result_full = np.triu(a_scaled_upper) + np.triu(a_scaled_upper, 1).T

        a_scaled_lower = mb01ss('D', 'L', a.copy(order='F'), d)
        result_full_lower = np.tril(a_scaled_lower) + np.tril(a_scaled_lower, -1).T

        np.testing.assert_allclose(result_full, result_full_lower, rtol=1e-14)

    def test_eigenvalue_scaling(self):
        """
        Verify eigenvalue scaling under congruence transformation.

        For positive definite A with eigenvalues lambda_i, the scaling
        D*A*D changes eigenvalues but preserves positive definiteness.

        Random seed: 300 (for reproducibility)
        """
        np.random.seed(300)
        n = 4
        a_full = np.random.randn(n, n)
        a = a_full @ a_full.T + np.eye(n)
        a = np.asfortranarray(a)

        eig_before = np.linalg.eigvalsh(a)
        assert np.all(eig_before > 0), "A should be positive definite"

        d = np.array([2.0, 3.0, 1.5, 0.5])
        a_scaled = mb01ss('D', 'U', a.copy(order='F'), d)

        a_scaled_full = np.triu(a_scaled) + np.triu(a_scaled, 1).T
        eig_after = np.linalg.eigvalsh(a_scaled_full)

        assert np.all(eig_after > 0), "Scaled A should remain positive definite"

    def test_identity_scaling(self):
        """
        Verify that scaling with D = ones preserves matrix.

        Random seed: 400 (for reproducibility)
        """
        np.random.seed(400)
        n = 3
        a_full = np.random.randn(n, n)
        a = (a_full + a_full.T) / 2
        a = np.asfortranarray(a)
        d = np.ones(n)

        a_scaled = mb01ss('D', 'U', a.copy(order='F'), d)

        np.testing.assert_allclose(np.triu(a_scaled), np.triu(a), rtol=1e-14)


class TestMB01SSEdgeCases:
    """Edge case tests."""

    def test_n_equals_zero(self):
        """Test with empty matrix (n=0)."""
        a = np.array([], dtype=float, order='F').reshape(0, 0)
        d = np.array([], dtype=float)

        a_result = mb01ss('D', 'U', a, d)

        assert a_result.shape == (0, 0)

    def test_n_equals_one(self):
        """Test with 1x1 matrix."""
        a = np.array([[5.0]], order='F')
        d = np.array([2.0])

        a_result = mb01ss('D', 'U', a.copy(order='F'), d)

        expected = 2.0 * 5.0 * 2.0
        np.testing.assert_allclose(a_result[0, 0], expected, rtol=1e-14)

    def test_n_equals_one_inv(self):
        """Test with 1x1 matrix using inverse scaling."""
        a = np.array([[8.0]], order='F')
        d = np.array([4.0])

        a_result = mb01ss('I', 'U', a.copy(order='F'), d)

        expected = (1.0/4.0) * 8.0 * (1.0/4.0)
        np.testing.assert_allclose(a_result[0, 0], expected, rtol=1e-14)

    def test_large_scaling_factors(self):
        """
        Test with large scaling factors to verify numerical stability.

        Random seed: 500 (for reproducibility)
        """
        np.random.seed(500)
        n = 3
        a_full = np.random.randn(n, n)
        a = (a_full + a_full.T) / 2
        a = np.asfortranarray(a)
        d = np.array([1e6, 1e-6, 1.0])

        a_result = mb01ss('D', 'U', a.copy(order='F'), d)

        d_mat = np.diag(d)
        expected = d_mat @ a @ d_mat

        np.testing.assert_allclose(
            np.triu(a_result), np.triu(expected), rtol=1e-10
        )


class TestMB01SSLowercase:
    """Test lowercase parameter handling."""

    def test_lowercase_jobs(self):
        """Test that lowercase 'd' works for JOBS parameter."""
        np.random.seed(600)
        n = 3
        a_full = np.random.randn(n, n)
        a = (a_full + a_full.T) / 2
        a = np.asfortranarray(a)
        d = np.array([2.0, 3.0, 0.5])

        a_upper = mb01ss('D', 'U', a.copy(order='F'), d)
        a_lower_case = mb01ss('d', 'U', a.copy(order='F'), d)

        np.testing.assert_allclose(a_upper, a_lower_case, rtol=1e-14)

    def test_lowercase_uplo(self):
        """Test that lowercase 'u' works for UPLO parameter."""
        np.random.seed(700)
        n = 3
        a_full = np.random.randn(n, n)
        a = (a_full + a_full.T) / 2
        a = np.asfortranarray(a)
        d = np.array([2.0, 3.0, 0.5])

        a_upper = mb01ss('D', 'U', a.copy(order='F'), d)
        a_lower_case = mb01ss('D', 'u', a.copy(order='F'), d)

        np.testing.assert_allclose(a_upper, a_lower_case, rtol=1e-14)
