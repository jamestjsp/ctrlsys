"""
Tests for MB01OH - Symmetric rank 2k update with two Hessenberg matrices.

Computes: R := alpha*R + beta*H*A' + beta*A*H' (trans='N')
      or: R := alpha*R + beta*H'*A + beta*A'*H (trans='T'/'C')

where R is symmetric, H and A are N-by-N upper Hessenberg matrices.
"""

import numpy as np
import pytest
from ctrlsys import mb01oh


def make_upper_hessenberg(a):
    """Zero out elements below subdiagonal."""
    n = a.shape[0]
    for j in range(n):
        for i in range(j + 2, n):
            a[i, j] = 0.0
    return a


def make_symmetric(a, uplo='U'):
    """Make matrix symmetric from upper or lower triangle."""
    n = a.shape[0]
    if uplo == 'U':
        for i in range(n):
            for j in range(i):
                a[i, j] = a[j, i]
    else:
        for i in range(n):
            for j in range(i + 1, n):
                a[i, j] = a[j, i]
    return a


class TestMB01OHBasic:
    """Basic functionality tests."""

    def test_notrans_upper_basic(self):
        """
        Test R := alpha*R + beta*H*A' + beta*A*H' with upper triangle.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n = 3
        alpha = 0.5
        beta = 1.0

        r = np.array([
            [1.0, 2.0, 3.0],
            [0.0, 4.0, 5.0],
            [0.0, 0.0, 6.0]
        ], order='F', dtype=float)

        h = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [0.0, 7.0, 8.0]
        ], order='F', dtype=float)

        a = np.array([
            [1.0, 1.0, 2.0],
            [3.0, 2.0, 1.0],
            [0.0, 4.0, 3.0]
        ], order='F', dtype=float)

        r_out, info = mb01oh('U', 'N', n, alpha, beta, r, h, a)

        assert info == 0

        h_full = h.copy()
        a_full = a.copy()
        r_sym = make_symmetric(np.array([
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 5.0],
            [3.0, 5.0, 6.0]
        ], order='F', dtype=float), 'U')

        expected_full = alpha * r_sym + beta * (h_full @ a_full.T) + beta * (a_full @ h_full.T)

        for j in range(n):
            for i in range(j + 1):
                np.testing.assert_allclose(r_out[i, j], expected_full[i, j], rtol=1e-14)

    def test_trans_upper_basic(self):
        """
        Test R := alpha*R + beta*H'*A + beta*A'*H with upper triangle.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n = 3
        alpha = 0.0
        beta = 1.0

        r = np.zeros((n, n), order='F', dtype=float)

        h = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [0.0, 7.0, 8.0]
        ], order='F', dtype=float)

        a = np.array([
            [1.0, 1.0, 2.0],
            [3.0, 2.0, 1.0],
            [0.0, 4.0, 3.0]
        ], order='F', dtype=float)

        r_out, info = mb01oh('U', 'T', n, alpha, beta, r, h, a)

        assert info == 0

        expected_full = beta * (h.T @ a) + beta * (a.T @ h)

        for j in range(n):
            for i in range(j + 1):
                np.testing.assert_allclose(r_out[i, j], expected_full[i, j], rtol=1e-14)

    def test_notrans_lower_basic(self):
        """
        Test R := alpha*R + beta*H*A' + beta*A*H' with lower triangle.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n = 3
        alpha = 1.0
        beta = 0.5

        r = np.array([
            [1.0, 0.0, 0.0],
            [2.0, 4.0, 0.0],
            [3.0, 5.0, 6.0]
        ], order='F', dtype=float)

        h = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [0.0, 7.0, 8.0]
        ], order='F', dtype=float)

        a = np.array([
            [1.0, 1.0, 2.0],
            [3.0, 2.0, 1.0],
            [0.0, 4.0, 3.0]
        ], order='F', dtype=float)

        r_out, info = mb01oh('L', 'N', n, alpha, beta, r, h, a)

        assert info == 0

        r_sym = make_symmetric(np.array([
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 5.0],
            [3.0, 5.0, 6.0]
        ], order='F', dtype=float), 'L')

        expected_full = alpha * r_sym + beta * (h @ a.T) + beta * (a @ h.T)

        for j in range(n):
            for i in range(j, n):
                np.testing.assert_allclose(r_out[i, j], expected_full[i, j], rtol=1e-14)


class TestMB01OHEdgeCases:
    """Edge case tests."""

    def test_n_zero(self):
        """Test with n=0 (quick return)."""
        r = np.zeros((1, 1), order='F', dtype=float)
        h = np.zeros((1, 1), order='F', dtype=float)
        a = np.zeros((1, 1), order='F', dtype=float)

        r_out, info = mb01oh('U', 'N', 0, 1.0, 1.0, r, h, a)
        assert info == 0

    def test_alpha_one_beta_zero(self):
        """Test alpha=1, beta=0 (R unchanged, quick return)."""
        n = 3
        r_orig = np.array([
            [1.0, 2.0, 3.0],
            [0.0, 4.0, 5.0],
            [0.0, 0.0, 6.0]
        ], order='F', dtype=float)
        r = r_orig.copy()
        h = np.eye(n, order='F', dtype=float)
        a = np.eye(n, order='F', dtype=float)

        r_out, info = mb01oh('U', 'N', n, 1.0, 0.0, r, h, a)

        assert info == 0
        for j in range(n):
            for i in range(j + 1):
                np.testing.assert_allclose(r_out[i, j], r_orig[i, j], rtol=1e-14)

    def test_alpha_zero_beta_nonzero(self):
        """
        Test alpha=0, beta!=0 (R zeroed first, then computed).

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n = 3
        alpha = 0.0
        beta = 2.0

        r = np.array([
            [9.0, 8.0, 7.0],
            [0.0, 6.0, 5.0],
            [0.0, 0.0, 4.0]
        ], order='F', dtype=float)

        h = np.array([
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 3.0, 1.0]
        ], order='F', dtype=float)

        a = np.array([
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [0.0, 2.0, 1.0]
        ], order='F', dtype=float)

        r_out, info = mb01oh('U', 'N', n, alpha, beta, r, h, a)

        assert info == 0

        expected_full = beta * (h @ a.T) + beta * (a @ h.T)

        for j in range(n):
            for i in range(j + 1):
                np.testing.assert_allclose(r_out[i, j], expected_full[i, j], rtol=1e-14)

    def test_alpha_zero_beta_zero(self):
        """Test alpha=0, beta=0 (R set to zero)."""
        n = 3
        r = np.array([
            [9.0, 8.0, 7.0],
            [0.0, 6.0, 5.0],
            [0.0, 0.0, 4.0]
        ], order='F', dtype=float)
        h = np.eye(n, order='F', dtype=float)
        a = np.eye(n, order='F', dtype=float)

        r_out, info = mb01oh('U', 'N', n, 0.0, 0.0, r, h, a)

        assert info == 0
        for j in range(n):
            for i in range(j + 1):
                np.testing.assert_allclose(r_out[i, j], 0.0, atol=1e-15)


class TestMB01OHErrors:
    """Error handling tests."""

    def test_invalid_uplo(self):
        """Test invalid UPLO parameter."""
        n = 3
        r = np.eye(n, order='F', dtype=float)
        h = np.eye(n, order='F', dtype=float)
        a = np.eye(n, order='F', dtype=float)

        r_out, info = mb01oh('X', 'N', n, 1.0, 1.0, r, h, a)
        assert info == -1

    def test_invalid_trans(self):
        """Test invalid TRANS parameter."""
        n = 3
        r = np.eye(n, order='F', dtype=float)
        h = np.eye(n, order='F', dtype=float)
        a = np.eye(n, order='F', dtype=float)

        r_out, info = mb01oh('U', 'X', n, 1.0, 1.0, r, h, a)
        assert info == -2

    def test_negative_n(self):
        """Test n < 0."""
        r = np.eye(3, order='F', dtype=float)
        h = np.eye(3, order='F', dtype=float)
        a = np.eye(3, order='F', dtype=float)

        r_out, info = mb01oh('U', 'N', -1, 1.0, 1.0, r, h, a)
        assert info == -3


class TestMB01OHMathematicalProperties:
    """Mathematical property validation tests."""

    def test_symmetry_of_result(self):
        """
        Verify the result R is symmetric.

        The operation R := alpha*R + beta*H*A' + beta*A*H' produces symmetric R.
        Random seed: 111 (for reproducibility)
        """
        np.random.seed(111)
        n = 4
        alpha = 0.0
        beta = 1.0

        r = np.zeros((n, n), order='F', dtype=float)

        h = make_upper_hessenberg(np.random.randn(n, n).astype(float, order='F'))
        a = make_upper_hessenberg(np.random.randn(n, n).astype(float, order='F'))

        r_out, info = mb01oh('U', 'N', n, alpha, beta, r, h, a)
        assert info == 0

        r_full = r_out.copy()
        for j in range(n):
            for i in range(j + 1):
                r_full[j, i] = r_full[i, j]

        expected = beta * (h @ a.T) + beta * (a @ h.T)

        np.testing.assert_allclose(r_full, expected, rtol=1e-14)

    def test_trans_equivalence(self):
        """
        Verify TRANS='T' and TRANS='C' produce same result.

        Random seed: 222 (for reproducibility)
        """
        np.random.seed(222)
        n = 4

        r1 = np.zeros((n, n), order='F', dtype=float)
        r2 = np.zeros((n, n), order='F', dtype=float)

        h = make_upper_hessenberg(np.random.randn(n, n).astype(float, order='F'))
        a = make_upper_hessenberg(np.random.randn(n, n).astype(float, order='F'))

        r_out1, info1 = mb01oh('U', 'T', n, 0.0, 1.0, r1, h.copy(), a.copy())
        r_out2, info2 = mb01oh('U', 'C', n, 0.0, 1.0, r2, h.copy(), a.copy())

        assert info1 == 0
        assert info2 == 0

        for j in range(n):
            for i in range(j + 1):
                np.testing.assert_allclose(r_out1[i, j], r_out2[i, j], rtol=1e-14)

    def test_scaling_property(self):
        """
        Verify scaling: R(2*beta) = 2*R(beta) when alpha=0.

        Random seed: 333 (for reproducibility)
        """
        np.random.seed(333)
        n = 4
        beta = 1.5

        h = make_upper_hessenberg(np.random.randn(n, n).astype(float, order='F'))
        a = make_upper_hessenberg(np.random.randn(n, n).astype(float, order='F'))

        r1 = np.zeros((n, n), order='F', dtype=float)
        r2 = np.zeros((n, n), order='F', dtype=float)

        r_out1, info1 = mb01oh('U', 'N', n, 0.0, beta, r1, h.copy(), a.copy())
        r_out2, info2 = mb01oh('U', 'N', n, 0.0, 2 * beta, r2, h.copy(), a.copy())

        assert info1 == 0
        assert info2 == 0

        for j in range(n):
            for i in range(j + 1):
                np.testing.assert_allclose(r_out2[i, j], 2.0 * r_out1[i, j], rtol=1e-14)

    def test_linearity_in_alpha(self):
        """
        Verify linearity in alpha.

        Random seed: 444 (for reproducibility)
        """
        np.random.seed(444)
        n = 3
        alpha1 = 0.5
        alpha2 = 1.0
        beta = 0.5

        r_init = np.array([
            [2.0, 4.0, 6.0],
            [0.0, 8.0, 10.0],
            [0.0, 0.0, 12.0]
        ], order='F', dtype=float)

        h = make_upper_hessenberg(np.array([
            [1.0, 2.0, 3.0],
            [1.0, 1.0, 2.0],
            [0.0, 1.0, 1.0]
        ], order='F', dtype=float))

        a = make_upper_hessenberg(np.array([
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [0.0, 2.0, 1.0]
        ], order='F', dtype=float))

        r_out1, info1 = mb01oh('U', 'N', n, alpha1, beta, r_init.copy(), h.copy(), a.copy())
        r_out2, info2 = mb01oh('U', 'N', n, alpha2, beta, r_init.copy(), h.copy(), a.copy())

        assert info1 == 0
        assert info2 == 0

        r_sym = make_symmetric(r_init.copy(), 'U')
        prod = beta * (h @ a.T + a @ h.T)

        expected1 = alpha1 * r_sym + prod
        expected2 = alpha2 * r_sym + prod

        for j in range(n):
            for i in range(j + 1):
                np.testing.assert_allclose(r_out1[i, j], expected1[i, j], rtol=1e-14)
                np.testing.assert_allclose(r_out2[i, j], expected2[i, j], rtol=1e-14)


class TestMB01OHLargerMatrix:
    """Test with larger matrices for more thorough validation."""

    def test_larger_matrix_notrans(self):
        """
        Test with larger 5x5 matrix, TRANS='N'.

        Random seed: 555 (for reproducibility)
        """
        np.random.seed(555)
        n = 5
        alpha = 0.3
        beta = 0.7

        r = np.triu(np.random.randn(n, n)).astype(float, order='F')
        r_sym = make_symmetric(r.copy(), 'U')
        h = make_upper_hessenberg(np.random.randn(n, n).astype(float, order='F'))
        a = make_upper_hessenberg(np.random.randn(n, n).astype(float, order='F'))

        r_out, info = mb01oh('U', 'N', n, alpha, beta, r.copy(), h, a)

        assert info == 0

        expected = alpha * r_sym + beta * (h @ a.T) + beta * (a @ h.T)

        for j in range(n):
            for i in range(j + 1):
                np.testing.assert_allclose(r_out[i, j], expected[i, j], rtol=1e-13)

    def test_larger_matrix_trans(self):
        """
        Test with larger 5x5 matrix, TRANS='T'.

        Random seed: 666 (for reproducibility)
        """
        np.random.seed(666)
        n = 5
        alpha = 0.2
        beta = 0.8

        r = np.triu(np.random.randn(n, n)).astype(float, order='F')
        r_sym = make_symmetric(r.copy(), 'U')
        h = make_upper_hessenberg(np.random.randn(n, n).astype(float, order='F'))
        a = make_upper_hessenberg(np.random.randn(n, n).astype(float, order='F'))

        r_out, info = mb01oh('U', 'T', n, alpha, beta, r.copy(), h, a)

        assert info == 0

        expected = alpha * r_sym + beta * (h.T @ a) + beta * (a.T @ h)

        for j in range(n):
            for i in range(j + 1):
                np.testing.assert_allclose(r_out[i, j], expected[i, j], rtol=1e-13)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
