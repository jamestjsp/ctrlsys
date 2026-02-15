"""
Tests for mb01oo - Compute P = op(H)*X*op(E)' or P' where H is upper Hessenberg,
X is symmetric, and E is upper triangular.

Test data generated using NumPy with deterministic seeds for reproducibility.
"""

import numpy as np
import pytest
from ctrlsys import mb01oo


def make_symmetric(a):
    """Make matrix symmetric by averaging with its transpose."""
    return (a + a.T) / 2


def make_upper_hessenberg(a):
    """Zero out elements below subdiagonal to make upper Hessenberg."""
    n = a.shape[0]
    h = a.copy()
    for i in range(2, n):
        for j in range(i - 1):
            h[i, j] = 0.0
    return h


class TestMB01OOBasic:
    """Basic functionality tests for mb01oo."""

    def test_hxe_trans_n(self):
        """
        Test P = H*X*E' with trans='N', uplo='U'.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n = 4

        h = make_upper_hessenberg(np.random.randn(n, n).astype(float, order='F'))
        x_full = make_symmetric(np.random.randn(n, n).astype(float, order='F'))
        x = np.triu(x_full).astype(float, order='F')
        e = np.triu(np.random.randn(n, n).astype(float, order='F'))

        p, info = mb01oo('U', 'N', h, x, e)

        assert info == 0
        assert p.shape == (n, n)

        p_expected = h @ x_full @ e.T
        np.testing.assert_allclose(p, p_expected, rtol=1e-14)

    def test_exh_trans_t(self):
        """
        Test P' = E'*X*H with trans='T', uplo='U'.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n = 4

        h = make_upper_hessenberg(np.random.randn(n, n).astype(float, order='F'))
        x_full = make_symmetric(np.random.randn(n, n).astype(float, order='F'))
        x = np.triu(x_full).astype(float, order='F')
        e = np.triu(np.random.randn(n, n).astype(float, order='F'))

        p, info = mb01oo('U', 'T', h, x, e)

        assert info == 0
        assert p.shape == (n, n)

        p_expected = e.T @ x_full @ h
        np.testing.assert_allclose(p, p_expected, rtol=1e-14)

    def test_hxe_uplo_l(self):
        """
        Test P = H*X*E' with trans='N', uplo='L'.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n = 4

        h = make_upper_hessenberg(np.random.randn(n, n).astype(float, order='F'))
        x_full = make_symmetric(np.random.randn(n, n).astype(float, order='F'))
        x = np.tril(x_full).astype(float, order='F')
        e = np.triu(np.random.randn(n, n).astype(float, order='F'))

        p, info = mb01oo('L', 'N', h, x, e)

        assert info == 0
        assert p.shape == (n, n)

        p_expected = h @ x_full @ e.T
        np.testing.assert_allclose(p, p_expected, rtol=1e-14)

    def test_exh_uplo_l(self):
        """
        Test P' = E'*X*H with trans='T', uplo='L'.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n = 4

        h = make_upper_hessenberg(np.random.randn(n, n).astype(float, order='F'))
        x_full = make_symmetric(np.random.randn(n, n).astype(float, order='F'))
        x = np.tril(x_full).astype(float, order='F')
        e = np.triu(np.random.randn(n, n).astype(float, order='F'))

        p, info = mb01oo('L', 'T', h, x, e)

        assert info == 0
        assert p.shape == (n, n)

        p_expected = e.T @ x_full @ h
        np.testing.assert_allclose(p, p_expected, rtol=1e-14)


class TestMB01OOEdgeCases:
    """Edge case tests for mb01oo."""

    def test_n_equals_1(self):
        """
        Test with 1x1 matrices.
        """
        h = np.array([[2.0]], order='F', dtype=float)
        x = np.array([[3.0]], order='F', dtype=float)
        e = np.array([[4.0]], order='F', dtype=float)

        p, info = mb01oo('U', 'N', h, x, e)

        assert info == 0
        assert p.shape == (1, 1)
        np.testing.assert_allclose(p[0, 0], 24.0, rtol=1e-14)

    def test_n_equals_2(self):
        """
        Test with 2x2 matrices.

        Random seed: 222 (for reproducibility)
        """
        np.random.seed(222)
        n = 2

        h = np.array([[1.0, 2.0], [3.0, 4.0]], order='F', dtype=float)
        x_full = np.array([[5.0, 6.0], [6.0, 7.0]], order='F', dtype=float)
        x = np.triu(x_full).astype(float, order='F')
        e = np.array([[2.0, 1.0], [0.0, 3.0]], order='F', dtype=float)

        p, info = mb01oo('U', 'N', h, x, e)

        assert info == 0
        p_expected = h @ x_full @ e.T
        np.testing.assert_allclose(p, p_expected, rtol=1e-14)

    def test_identity_matrices(self):
        """
        Test P = I*X*I' = X with identity Hessenberg and E.

        Random seed: 333 (for reproducibility)
        """
        np.random.seed(333)
        n = 3

        h = np.eye(n, order='F', dtype=float)
        x_full = make_symmetric(np.random.randn(n, n).astype(float, order='F'))
        x = np.triu(x_full).astype(float, order='F')
        e = np.eye(n, order='F', dtype=float)

        p, info = mb01oo('U', 'N', h, x, e)

        assert info == 0
        np.testing.assert_allclose(p, x_full, rtol=1e-14)

    def test_zero_matrices(self):
        """
        Test with zero X matrix.
        """
        n = 3

        h = make_upper_hessenberg(np.ones((n, n), order='F', dtype=float))
        x = np.zeros((n, n), order='F', dtype=float)
        e = np.triu(np.ones((n, n), order='F', dtype=float))

        p, info = mb01oo('U', 'N', h, x, e)

        assert info == 0
        np.testing.assert_allclose(p, np.zeros((n, n)), rtol=1e-14)


class TestMB01OOProperties:
    """Mathematical property tests for mb01oo."""

    def test_relationship_trans_n_t(self):
        """
        Test relationship: trans='N' computes P=H*X*E', trans='T' computes P'=E'*X*H.

        Since P = H*X*E', we have P' = E*X'*H' = E*X*H' (X symmetric).
        The routine with trans='T' computes P' = E'*X*H.
        So P_N' (from trans='N') should equal the expected P' = E*X*H'.
        But trans='T' computes E'*X*H, which is different.

        Actually the documentation states:
        - trans='N': output is P = H*X*E'
        - trans='T': output is P' = E'*X*H

        So the output from trans='T' is P' (not P), meaning the actual matrix P
        would be P = (E'*X*H)' = H'*X*E.

        This test verifies both compute valid results and cross-validates.

        Random seed: 444 (for reproducibility)
        """
        np.random.seed(444)
        n = 4

        h = make_upper_hessenberg(np.random.randn(n, n).astype(float, order='F'))
        x_full = make_symmetric(np.random.randn(n, n).astype(float, order='F'))
        x_u = np.triu(x_full).astype(float, order='F')
        e = np.triu(np.random.randn(n, n).astype(float, order='F'))

        p_n, info_n = mb01oo('U', 'N', h, x_u, e)
        p_t, info_t = mb01oo('U', 'T', h, x_u, e)

        assert info_n == 0
        assert info_t == 0

        p_expected_n = h @ x_full @ e.T
        p_prime_expected = e.T @ x_full @ h

        np.testing.assert_allclose(p_n, p_expected_n, rtol=1e-14)
        np.testing.assert_allclose(p_t, p_prime_expected, rtol=1e-14)

    def test_linearity_in_x(self):
        """
        Test linearity: H*(alpha*X)*E' = alpha*(H*X*E').

        Random seed: 555 (for reproducibility)
        """
        np.random.seed(555)
        n = 4
        alpha = 2.5

        h = make_upper_hessenberg(np.random.randn(n, n).astype(float, order='F'))
        x_full = make_symmetric(np.random.randn(n, n).astype(float, order='F'))
        x = np.triu(x_full).astype(float, order='F')
        x_scaled = np.triu(alpha * x_full).astype(float, order='F')
        e = np.triu(np.random.randn(n, n).astype(float, order='F'))

        p1, info1 = mb01oo('U', 'N', h, x_scaled, e)
        p2, info2 = mb01oo('U', 'N', h, x, e)

        assert info1 == 0
        assert info2 == 0
        np.testing.assert_allclose(p1, alpha * p2, rtol=1e-14)

    def test_consistency_upper_lower(self):
        """
        Test that upper and lower storage produce same result.

        Random seed: 666 (for reproducibility)
        """
        np.random.seed(666)
        n = 4

        h = make_upper_hessenberg(np.random.randn(n, n).astype(float, order='F'))
        x_full = make_symmetric(np.random.randn(n, n).astype(float, order='F'))
        x_upper = np.triu(x_full).astype(float, order='F')
        x_lower = np.tril(x_full).astype(float, order='F')
        e = np.triu(np.random.randn(n, n).astype(float, order='F'))

        p_upper, info_u = mb01oo('U', 'N', h, x_upper, e)
        p_lower, info_l = mb01oo('L', 'N', h, x_lower, e)

        assert info_u == 0
        assert info_l == 0
        np.testing.assert_allclose(p_upper, p_lower, rtol=1e-14)

    def test_larger_matrix(self):
        """
        Test with larger matrix to ensure algorithm scales.

        Random seed: 777 (for reproducibility)
        """
        np.random.seed(777)
        n = 10

        h = make_upper_hessenberg(np.random.randn(n, n).astype(float, order='F'))
        x_full = make_symmetric(np.random.randn(n, n).astype(float, order='F'))
        x = np.triu(x_full).astype(float, order='F')
        e = np.triu(np.random.randn(n, n).astype(float, order='F'))

        p, info = mb01oo('U', 'N', h, x, e)

        assert info == 0
        p_expected = h @ x_full @ e.T
        np.testing.assert_allclose(p, p_expected, rtol=1e-13)


class TestMB01OOErrors:
    """Error handling tests for mb01oo."""

    def test_invalid_uplo(self):
        """Test that invalid uplo returns error."""
        n = 3
        h = np.zeros((n, n), order='F', dtype=float)
        x = np.zeros((n, n), order='F', dtype=float)
        e = np.zeros((n, n), order='F', dtype=float)

        p, info = mb01oo('X', 'N', h, x, e)

        assert info == -1

    def test_invalid_trans(self):
        """Test that invalid trans returns error."""
        n = 3
        h = np.zeros((n, n), order='F', dtype=float)
        x = np.zeros((n, n), order='F', dtype=float)
        e = np.zeros((n, n), order='F', dtype=float)

        p, info = mb01oo('U', 'X', h, x, e)

        assert info == -2
