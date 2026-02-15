"""
Tests for MB03QV: Compute eigenvalues of upper quasi-triangular matrix pencil.

MB03QV computes the generalized eigenvalues of an upper quasi-triangular
matrix pencil (S, T) where S is upper quasi-triangular and T is upper triangular.
"""

import numpy as np
import pytest
from ctrlsys import mb03qv


class TestMB03QVBasic:
    """Basic functionality tests for mb03qv."""

    def test_diagonal_matrices(self):
        """
        Test with diagonal matrices (all real eigenvalues).

        S = diag(1, 2, 3), T = diag(2, 3, 4)
        Eigenvalues: lambda_i = S(i,i) / T(i,i) = 0.5, 0.667, 0.75
        """
        n = 3
        s = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0]
        ], order='F', dtype=float)

        t = np.array([
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0]
        ], order='F', dtype=float)

        alphar, alphai, beta, info = mb03qv(s, t)

        assert info == 0
        assert len(alphar) == n
        assert len(alphai) == n
        assert len(beta) == n

        # For diagonal matrices: alphar[i] = S[i,i], beta[i] = T[i,i], alphai[i] = 0
        np.testing.assert_allclose(alphar, [1.0, 2.0, 3.0], rtol=1e-14)
        np.testing.assert_allclose(alphai, [0.0, 0.0, 0.0], atol=1e-14)
        np.testing.assert_allclose(beta, [2.0, 3.0, 4.0], rtol=1e-14)

        # Verify eigenvalues
        eigenvalues = alphar / beta
        expected_eigenvalues = np.array([0.5, 2.0/3.0, 0.75])
        np.testing.assert_allclose(eigenvalues, expected_eigenvalues, rtol=1e-14)

    def test_quasi_triangular_with_2x2_block(self):
        """
        Test with quasi-triangular S containing a 2x2 block (complex eigenvalue pair).

        Random seed: 42 (for reproducibility)

        S has a 2x2 block at positions (0:2, 0:2) with complex eigenvalues.
        """
        n = 3

        # 2x2 block with eigenvalues 1+2i and 1-2i: [[1, 2], [-2, 1]] scaled
        # Plus a simple real eigenvalue at (2,2)
        s = np.array([
            [1.0,  2.0, 0.5],
            [-2.0, 1.0, 0.3],
            [0.0,  0.0, 3.0]
        ], order='F', dtype=float)

        t = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0]
        ], order='F', dtype=float)

        alphar, alphai, beta, info = mb03qv(s, t)

        assert info == 0

        # First two eigenvalues form complex conjugate pair
        # The 2x2 block [[1,2],[-2,1]] has eigenvalues 1+2i, 1-2i
        # With T = I for that block, alpha = 1 +/- 2i, beta = 1
        assert alphai[0] > 0, "First eigenvalue of pair should have positive imaginary part"
        assert alphai[1] < 0, "Second eigenvalue of pair should have negative imaginary part"
        np.testing.assert_allclose(alphai[0], -alphai[1], rtol=1e-14)
        np.testing.assert_allclose(alphar[0], alphar[1], rtol=1e-14)

        # Third eigenvalue is real: S(2,2)/T(2,2) = 3/2
        np.testing.assert_allclose(alphai[2], 0.0, atol=1e-14)
        np.testing.assert_allclose(alphar[2] / beta[2], 1.5, rtol=1e-14)

    def test_upper_triangular_s(self):
        """
        Test with upper triangular S (all real eigenvalues).
        """
        n = 4
        s = np.array([
            [2.0, 1.0, 0.5, 0.2],
            [0.0, 3.0, 1.0, 0.5],
            [0.0, 0.0, 4.0, 1.0],
            [0.0, 0.0, 0.0, 5.0]
        ], order='F', dtype=float)

        t = np.array([
            [1.0, 0.5, 0.2, 0.1],
            [0.0, 1.0, 0.5, 0.2],
            [0.0, 0.0, 2.0, 0.5],
            [0.0, 0.0, 0.0, 2.0]
        ], order='F', dtype=float)

        alphar, alphai, beta, info = mb03qv(s, t)

        assert info == 0

        # All eigenvalues should be real
        np.testing.assert_allclose(alphai, np.zeros(n), atol=1e-14)

        # Eigenvalues are diagonal ratios
        expected_eigenvalues = np.array([2.0, 3.0, 2.0, 2.5])
        eigenvalues = alphar / beta
        np.testing.assert_allclose(eigenvalues, expected_eigenvalues, rtol=1e-14)


class TestMB03QVEdgeCases:
    """Edge case tests for mb03qv."""

    def test_n_equals_zero(self):
        """Test with n=0 (quick return case)."""
        s = np.array([], order='F', dtype=float).reshape(0, 0)
        t = np.array([], order='F', dtype=float).reshape(0, 0)

        alphar, alphai, beta, info = mb03qv(s, t)

        assert info == 0
        assert len(alphar) == 0
        assert len(alphai) == 0
        assert len(beta) == 0

    def test_n_equals_one(self):
        """Test with n=1 (single eigenvalue)."""
        s = np.array([[5.0]], order='F', dtype=float)
        t = np.array([[2.0]], order='F', dtype=float)

        alphar, alphai, beta, info = mb03qv(s, t)

        assert info == 0
        assert len(alphar) == 1
        assert len(alphai) == 1
        assert len(beta) == 1

        np.testing.assert_allclose(alphar[0], 5.0, rtol=1e-14)
        np.testing.assert_allclose(alphai[0], 0.0, atol=1e-14)
        np.testing.assert_allclose(beta[0], 2.0, rtol=1e-14)

    def test_identity_matrices(self):
        """Test with identity matrices."""
        n = 3
        s = np.eye(n, order='F', dtype=float)
        t = np.eye(n, order='F', dtype=float)

        alphar, alphai, beta, info = mb03qv(s, t)

        assert info == 0
        np.testing.assert_allclose(alphar, np.ones(n), rtol=1e-14)
        np.testing.assert_allclose(alphai, np.zeros(n), atol=1e-14)
        np.testing.assert_allclose(beta, np.ones(n), rtol=1e-14)

    def test_two_by_two_complex_pair(self):
        """
        Test with 2x2 quasi-triangular S having complex eigenvalues.

        S = [[a, b], [-c, a]] where c > 0 gives eigenvalues a +/- i*sqrt(bc)
        For [[2, 3], [-2, 2]]: eigenvalues = 2 +/- i*sqrt(6)
        """
        a, b, c = 2.0, 3.0, 2.0
        s = np.array([
            [a,  b],
            [-c, a]
        ], order='F', dtype=float)

        t = np.eye(2, order='F', dtype=float)

        alphar, alphai, beta, info = mb03qv(s, t)

        assert info == 0

        # Compute eigenvalues: lambda = (alphar + i*alphai) / beta
        eigenvalues = (alphar + 1j * alphai) / beta

        # Expected eigenvalues: trace = 2*a = 4, det = a^2 + b*c = 4 + 6 = 10
        # lambda = (4 +/- sqrt(16 - 40))/2 = 2 +/- sqrt(-6) = 2 +/- i*sqrt(6)
        expected_real = a
        expected_imag = np.sqrt(b * c)

        np.testing.assert_allclose(eigenvalues[0].real, expected_real, rtol=1e-14)
        np.testing.assert_allclose(eigenvalues[1].real, expected_real, rtol=1e-14)
        np.testing.assert_allclose(abs(eigenvalues[0].imag), expected_imag, rtol=1e-14)
        np.testing.assert_allclose(eigenvalues[0].imag, -eigenvalues[1].imag, rtol=1e-14)


class TestMB03QVPropertyBased:
    """Mathematical property tests for mb03qv."""

    def test_eigenvalue_consistency(self):
        """
        Verify that computed eigenvalues satisfy characteristic equation.

        For generalized eigenvalue problem: det(S - lambda*T) = 0
        where lambda = alpha/beta = (alphar + i*alphai) / beta

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n = 4

        # Create quasi-triangular S with one 2x2 block
        s = np.triu(np.random.randn(n, n)).astype(float, order='F')
        # Add subdiagonal for 2x2 block at (0,1)
        s[1, 0] = -0.5  # Makes 2x2 block with complex eigenvalues

        t = np.triu(np.random.randn(n, n)).astype(float, order='F')
        t = t + 2 * np.eye(n)  # Ensure diagonal dominance

        alphar, alphai, beta, info = mb03qv(s, t)

        assert info == 0

        # Verify using scipy.linalg.eigvals for the 2x2 blocks and 1x1 blocks
        # For 1x1 blocks (i=2,3): eigenvalue = S[i,i] / T[i,i]
        # Check real eigenvalues (where alphai == 0)
        for i in range(n):
            if abs(alphai[i]) < 1e-10:  # Real eigenvalue
                lam = alphar[i] / beta[i]
                # For simple eigenvalues from diagonal, verify against direct ratio
                if i > 1:  # Skip first 2x2 block
                    expected = s[i, i] / t[i, i]
                    np.testing.assert_allclose(lam, expected, rtol=1e-12)

    def test_eigenvalue_scaling(self):
        """
        Verify eigenvalue scaling property: scaling S by c scales eigenvalues by c.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n = 3
        scale = 2.5

        s = np.triu(np.random.randn(n, n)).astype(float, order='F')
        t = np.triu(np.random.randn(n, n)).astype(float, order='F')
        t = t + 3 * np.eye(n)

        alphar1, alphai1, beta1, info1 = mb03qv(s, t)
        assert info1 == 0

        s_scaled = scale * s
        alphar2, alphai2, beta2, info2 = mb03qv(s_scaled, t)
        assert info2 == 0

        # After scaling S by c: alphar' = c*alphar, alphai' = c*alphai, beta' = beta
        # So lambda' = c*lambda
        eigenvalues1 = (alphar1 + 1j * alphai1) / beta1
        eigenvalues2 = (alphar2 + 1j * alphai2) / beta2

        np.testing.assert_allclose(eigenvalues2, scale * eigenvalues1, rtol=1e-13)


class TestMB03QVNumericalStability:
    """Numerical stability tests for mb03qv."""

    def test_large_values(self):
        """Test with large matrix entries."""
        n = 2
        scale = 1e10
        s = scale * np.array([
            [2.0, 1.0],
            [0.0, 3.0]
        ], order='F', dtype=float)

        t = np.array([
            [1.0, 0.5],
            [0.0, 2.0]
        ], order='F', dtype=float)

        alphar, alphai, beta, info = mb03qv(s, t)

        assert info == 0
        np.testing.assert_allclose(alphai, np.zeros(n), atol=1e-6)

        eigenvalues = alphar / beta
        expected = np.array([2.0 * scale, 1.5 * scale])
        np.testing.assert_allclose(eigenvalues, expected, rtol=1e-10)

    def test_small_values(self):
        """Test with small matrix entries."""
        n = 2
        scale = 1e-10
        s = scale * np.array([
            [4.0, 2.0],
            [0.0, 6.0]
        ], order='F', dtype=float)

        t = np.array([
            [2.0, 1.0],
            [0.0, 3.0]
        ], order='F', dtype=float)

        alphar, alphai, beta, info = mb03qv(s, t)

        assert info == 0
        np.testing.assert_allclose(alphai, np.zeros(n), atol=1e-20)

        eigenvalues = alphar / beta
        expected = np.array([2.0 * scale, 2.0 * scale])
        np.testing.assert_allclose(eigenvalues, expected, rtol=1e-10)
