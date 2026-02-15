"""
Tests for mb01os - Compute P = H*X or P = X*H where H is upper Hessenberg
and X is symmetric.

Test data generated using NumPy with deterministic seeds for reproducibility.
"""

import numpy as np
import pytest
from ctrlsys import mb01os


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


class TestMB01OSBasic:
    """Basic functionality tests for mb01os."""

    def test_hx_uplo_u_trans_n(self):
        """
        Test P = H*X with X upper triangular stored, trans='N'.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n = 4

        h = make_upper_hessenberg(np.random.randn(n, n).astype(float, order='F'))
        x_full = make_symmetric(np.random.randn(n, n).astype(float, order='F'))
        x = np.triu(x_full).astype(float, order='F')

        p, info = mb01os('U', 'N', h, x)

        assert info == 0
        assert p.shape == (n, n)

        p_expected = h @ x_full
        np.testing.assert_allclose(p, p_expected, rtol=1e-14)

    def test_xh_uplo_u_trans_t(self):
        """
        Test P = X*H with X upper triangular stored, trans='T'.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n = 4

        h = make_upper_hessenberg(np.random.randn(n, n).astype(float, order='F'))
        x_full = make_symmetric(np.random.randn(n, n).astype(float, order='F'))
        x = np.triu(x_full).astype(float, order='F')

        p, info = mb01os('U', 'T', h, x)

        assert info == 0
        assert p.shape == (n, n)

        p_expected = x_full @ h
        np.testing.assert_allclose(p, p_expected, rtol=1e-14)

    def test_hx_uplo_l_trans_n(self):
        """
        Test P = H*X with X lower triangular stored, trans='N'.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n = 4

        h = make_upper_hessenberg(np.random.randn(n, n).astype(float, order='F'))
        x_full = make_symmetric(np.random.randn(n, n).astype(float, order='F'))
        x = np.tril(x_full).astype(float, order='F')

        p, info = mb01os('L', 'N', h, x)

        assert info == 0
        assert p.shape == (n, n)

        p_expected = h @ x_full
        np.testing.assert_allclose(p, p_expected, rtol=1e-14)

    def test_xh_uplo_l_trans_t(self):
        """
        Test P = X*H with X lower triangular stored, trans='T'.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n = 4

        h = make_upper_hessenberg(np.random.randn(n, n).astype(float, order='F'))
        x_full = make_symmetric(np.random.randn(n, n).astype(float, order='F'))
        x = np.tril(x_full).astype(float, order='F')

        p, info = mb01os('L', 'T', h, x)

        assert info == 0
        assert p.shape == (n, n)

        p_expected = x_full @ h
        np.testing.assert_allclose(p, p_expected, rtol=1e-14)


class TestMB01OSEdgeCases:
    """Edge case tests for mb01os."""

    def test_n_equals_1(self):
        """
        Test with 1x1 matrices.

        Random seed: 111 (for reproducibility)
        """
        np.random.seed(111)
        n = 1

        h = np.array([[2.5]], order='F', dtype=float)
        x = np.array([[3.0]], order='F', dtype=float)

        p, info = mb01os('U', 'N', h, x)

        assert info == 0
        assert p.shape == (1, 1)
        np.testing.assert_allclose(p[0, 0], 7.5, rtol=1e-14)

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

        p, info = mb01os('U', 'N', h, x)

        assert info == 0
        p_expected = h @ x_full
        np.testing.assert_allclose(p, p_expected, rtol=1e-14)

    def test_identity_hessenberg(self):
        """
        Test P = I*X = X with identity Hessenberg matrix.

        Random seed: 333 (for reproducibility)
        """
        np.random.seed(333)
        n = 3

        h = np.eye(n, order='F', dtype=float)
        x_full = make_symmetric(np.random.randn(n, n).astype(float, order='F'))
        x = np.triu(x_full).astype(float, order='F')

        p, info = mb01os('U', 'N', h, x)

        assert info == 0
        np.testing.assert_allclose(p, x_full, rtol=1e-14)

    def test_zero_matrices(self):
        """
        Test with zero matrices.
        """
        n = 3

        h = np.zeros((n, n), order='F', dtype=float)
        x = np.zeros((n, n), order='F', dtype=float)

        p, info = mb01os('U', 'N', h, x)

        assert info == 0
        np.testing.assert_allclose(p, np.zeros((n, n)), rtol=1e-14)


class TestMB01OSProperties:
    """Mathematical property tests for mb01os."""

    def test_linearity_in_h(self):
        """
        Test linearity: (alpha*H)*X = alpha*(H*X).

        Random seed: 444 (for reproducibility)
        """
        np.random.seed(444)
        n = 4
        alpha = 2.5

        h = make_upper_hessenberg(np.random.randn(n, n).astype(float, order='F'))
        x_full = make_symmetric(np.random.randn(n, n).astype(float, order='F'))
        x = np.triu(x_full).astype(float, order='F')

        p1, info1 = mb01os('U', 'N', alpha * h, x)
        p2, info2 = mb01os('U', 'N', h, x)

        assert info1 == 0
        assert info2 == 0
        np.testing.assert_allclose(p1, alpha * p2, rtol=1e-14)

    def test_linearity_in_x(self):
        """
        Test linearity: H*(alpha*X) = alpha*(H*X).

        Random seed: 555 (for reproducibility)
        """
        np.random.seed(555)
        n = 4
        alpha = 3.0

        h = make_upper_hessenberg(np.random.randn(n, n).astype(float, order='F'))
        x_full = make_symmetric(np.random.randn(n, n).astype(float, order='F'))
        x = np.triu(x_full).astype(float, order='F')
        x_scaled = np.triu(alpha * x_full).astype(float, order='F')

        p1, info1 = mb01os('U', 'N', h, x_scaled)
        p2, info2 = mb01os('U', 'N', h, x)

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

        p_upper, info_u = mb01os('U', 'N', h, x_upper)
        p_lower, info_l = mb01os('L', 'N', h, x_lower)

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

        p, info = mb01os('U', 'N', h, x)

        assert info == 0
        p_expected = h @ x_full
        np.testing.assert_allclose(p, p_expected, rtol=1e-13)


class TestMB01OSErrors:
    """Error handling tests for mb01os."""

    def test_invalid_uplo(self):
        """Test that invalid uplo returns error."""
        n = 3
        h = np.zeros((n, n), order='F', dtype=float)
        x = np.zeros((n, n), order='F', dtype=float)

        p, info = mb01os('X', 'N', h, x)

        assert info == -1

    def test_invalid_trans(self):
        """Test that invalid trans returns error."""
        n = 3
        h = np.zeros((n, n), order='F', dtype=float)
        x = np.zeros((n, n), order='F', dtype=float)

        p, info = mb01os('U', 'X', h, x)

        assert info == -2
