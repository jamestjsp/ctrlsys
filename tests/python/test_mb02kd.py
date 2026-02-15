"""
Tests for MB02KD - Block Toeplitz matrix-matrix product.

Computes C = alpha*op(T)*B + beta*C where T is a block Toeplitz matrix.
"""

import numpy as np
import pytest
from ctrlsys import mb02kd


class TestMB02KDBasic:
    """Basic functionality tests from HTML documentation example."""

    def test_html_doc_example(self):
        """
        Test from MB02KD HTML doc example.

        K=3, L=2, M=4, N=5, R=1
        LDBLK='C' (T(1,1) in first block of TC)
        TRANS='N' (op(T) = T)
        alpha=1.0, beta=0.0

        Computes C = T * B where T is 12x10 block Toeplitz matrix.
        """
        k, l, m, n, r = 3, 2, 4, 5, 1

        tc = np.array([
            [4.0, 1.0],
            [3.0, 5.0],
            [2.0, 1.0],
            [4.0, 1.0],
            [3.0, 4.0],
            [2.0, 4.0],
            [3.0, 1.0],
            [3.0, 0.0],
            [4.0, 4.0],
            [5.0, 1.0],
            [3.0, 1.0],
            [4.0, 3.0],
        ], order='F', dtype=float)

        tr = np.array([
            [5.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 3.0],
            [4.0, 1.0, 5.0, 4.0, 5.0, 4.0, 1.0, 2.0],
            [2.0, 3.0, 4.0, 1.0, 3.0, 3.0, 3.0, 3.0],
        ], order='F', dtype=float)

        b = np.array([
            [0.0],
            [2.0],
            [2.0],
            [2.0],
            [1.0],
            [3.0],
            [3.0],
            [4.0],
            [2.0],
            [3.0],
        ], order='F', dtype=float)

        c_expected = np.array([
            [45.0],
            [76.0],
            [55.0],
            [44.0],
            [84.0],
            [56.0],
            [52.0],
            [70.0],
            [54.0],
            [49.0],
            [63.0],
            [59.0],
        ], order='F', dtype=float)

        c, info = mb02kd('C', 'N', k, l, m, n, r, 1.0, 0.0, tc, tr, b)

        assert info == 0
        np.testing.assert_allclose(c, c_expected, rtol=1e-10)


class TestMB02KDTranspose:
    """Tests for transpose operation (TRANS='T')."""

    def test_transpose_small(self):
        """
        Test with TRANS='T': C = T' * B.

        For K=2, L=2, M=2, N=2, R=1:
        T is 4x4, T' is 4x4
        B is 4x1, C is 4x1

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        k, l, m, n, r = 2, 2, 2, 2, 1

        tc = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ], order='F', dtype=float)

        tr = np.array([
            [9.0, 10.0],
            [11.0, 12.0],
        ], order='F', dtype=float)

        b = np.array([
            [1.0],
            [2.0],
            [3.0],
            [4.0],
        ], order='F', dtype=float)

        t = np.zeros((m * k, n * l), order='F', dtype=float)
        t[0:k, 0:l] = tc[0:k, :]
        t[k:2*k, 0:l] = tc[k:2*k, :]
        t[0:k, l:2*l] = tr[0:k, :]
        t[k:2*k, l:2*l] = tc[0:k, :]

        c_expected = t.T @ b

        c, info = mb02kd('C', 'T', k, l, m, n, r, 1.0, 0.0, tc, tr, b)

        assert info == 0
        np.testing.assert_allclose(c, c_expected, rtol=1e-10)


class TestMB02KDLdblkR:
    """Tests for LDBLK='R' (T(1,1) in first block of TR)."""

    def test_ldblk_r_simple(self):
        """
        Test with LDBLK='R': T(1,1)-block stored in first block of TR.

        K=2, L=2, M=2, N=2, R=1
        TC contains blocks 2..M of first block column (1 block = 2 rows)
        TR contains full first block row (2 blocks = 4 cols)
        """
        k, l, m, n, r = 2, 2, 2, 2, 1

        tc = np.array([
            [5.0, 6.0],
            [7.0, 8.0],
        ], order='F', dtype=float)

        tr = np.array([
            [1.0, 2.0, 9.0, 10.0],
            [3.0, 4.0, 11.0, 12.0],
        ], order='F', dtype=float)

        b = np.array([
            [1.0],
            [1.0],
            [1.0],
            [1.0],
        ], order='F', dtype=float)

        t = np.zeros((m * k, n * l), order='F', dtype=float)
        t[0:k, 0:l] = tr[0:k, 0:l]
        t[k:2*k, 0:l] = tc[0:k, :]
        t[0:k, l:2*l] = tr[0:k, l:2*l]
        t[k:2*k, l:2*l] = tr[0:k, 0:l]

        c_expected = t @ b

        c, info = mb02kd('R', 'N', k, l, m, n, r, 1.0, 0.0, tc, tr, b)

        assert info == 0
        np.testing.assert_allclose(c, c_expected, rtol=1e-10)


class TestMB02KDScalars:
    """Tests for alpha and beta scalars."""

    def test_alpha_beta(self):
        """
        Test C = alpha*T*B + beta*C with non-trivial alpha and beta.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        k, l, m, n, r = 2, 2, 2, 2, 1
        alpha, beta = 2.0, 0.5

        tc = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.0],
            [0.0, 0.5],
        ], order='F', dtype=float)

        tr = np.array([
            [0.25, 0.0],
            [0.0, 0.25],
        ], order='F', dtype=float)

        b = np.array([
            [1.0],
            [2.0],
            [3.0],
            [4.0],
        ], order='F', dtype=float)

        c_init = np.array([
            [10.0],
            [20.0],
            [30.0],
            [40.0],
        ], order='F', dtype=float)

        t = np.zeros((m * k, n * l), order='F', dtype=float)
        t[0:k, 0:l] = tc[0:k, :]
        t[k:2*k, 0:l] = tc[k:2*k, :]
        t[0:k, l:2*l] = tr[0:k, :]
        t[k:2*k, l:2*l] = tc[0:k, :]

        c_expected = alpha * (t @ b) + beta * c_init

        c, info = mb02kd('C', 'N', k, l, m, n, r, alpha, beta, tc, tr, b, c_init.copy())

        assert info == 0
        np.testing.assert_allclose(c, c_expected, rtol=1e-10)

    def test_alpha_zero(self):
        """
        Test with alpha=0: C = beta*C.

        When alpha=0, TC/TR/B should not be referenced.
        """
        k, l, m, n, r = 2, 2, 2, 2, 1
        alpha, beta = 0.0, 2.0

        tc = np.ones((m * k, l), order='F', dtype=float)
        tr = np.ones((k, (n - 1) * l), order='F', dtype=float)
        b = np.ones((n * l, r), order='F', dtype=float)
        c_init = np.array([
            [1.0],
            [2.0],
            [3.0],
            [4.0],
        ], order='F', dtype=float)

        c_expected = beta * c_init

        c, info = mb02kd('C', 'N', k, l, m, n, r, alpha, beta, tc, tr, b, c_init.copy())

        assert info == 0
        np.testing.assert_allclose(c, c_expected, rtol=1e-14)


class TestMB02KDEdgeCases:
    """Edge case tests."""

    def test_zero_dimensions(self):
        """Test with zero dimensions (quick return)."""
        tc = np.zeros((1, 1), order='F', dtype=float)
        tr = np.zeros((1, 1), order='F', dtype=float)
        b = np.zeros((1, 1), order='F', dtype=float)

        c, info = mb02kd('C', 'N', 0, 0, 0, 0, 0, 1.0, 0.0, tc, tr, b)

        assert info == 0

    def test_single_block(self):
        """
        Test with single block (M=1, N=1).

        T is just a K x L matrix.
        """
        k, l, r = 3, 2, 1

        tc = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ], order='F', dtype=float)

        tr = np.zeros((k, 0), order='F', dtype=float)

        b = np.array([
            [1.0],
            [1.0],
        ], order='F', dtype=float)

        c_expected = tc @ b

        c, info = mb02kd('C', 'N', k, l, 1, 1, r, 1.0, 0.0, tc, tr, b)

        assert info == 0
        np.testing.assert_allclose(c, c_expected, rtol=1e-14)


class TestMB02KDPropertyBased:
    """Property-based tests for mathematical correctness."""

    def test_linearity_in_b(self):
        """
        Verify linearity: T*(a*B1 + b*B2) = a*T*B1 + b*T*B2.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        k, l, m, n, r = 2, 2, 3, 3, 2
        a_scalar, b_scalar = 2.0, 3.0

        tc = np.random.randn(m * k, l).astype(float, order='F')
        tr = np.random.randn(k, (n - 1) * l).astype(float, order='F')
        b1 = np.random.randn(n * l, r).astype(float, order='F')
        b2 = np.random.randn(n * l, r).astype(float, order='F')

        b_combined = a_scalar * b1 + b_scalar * b2

        c1, info1 = mb02kd('C', 'N', k, l, m, n, r, 1.0, 0.0, tc, tr, b1)
        c2, info2 = mb02kd('C', 'N', k, l, m, n, r, 1.0, 0.0, tc, tr, b2)
        c_combined, info3 = mb02kd('C', 'N', k, l, m, n, r, 1.0, 0.0, tc, tr, b_combined)

        assert info1 == 0 and info2 == 0 and info3 == 0

        c_expected = a_scalar * c1 + b_scalar * c2
        np.testing.assert_allclose(c_combined, c_expected, rtol=1e-13)

    def test_transpose_consistency(self):
        """
        Verify transpose consistency: (T*B)' * e = B' * T' * e.

        Using the property: u' * (T*B) = (B' * T') * u' for any vector u.
        We verify via inner products that both computations are consistent.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        k, l, m, n, r = 2, 2, 3, 3, 2

        tc = np.random.randn(m * k, l).astype(float, order='F')
        tr = np.random.randn(k, (n - 1) * l).astype(float, order='F')
        b = np.random.randn(n * l, r).astype(float, order='F')
        e = np.random.randn(m * k, 1).astype(float, order='F')

        c_forward, info1 = mb02kd('C', 'N', k, l, m, n, r, 1.0, 0.0, tc, tr, b)
        assert info1 == 0

        d_transpose, info2 = mb02kd('C', 'T', k, l, m, n, 1, 1.0, 0.0, tc, tr, e)
        assert info2 == 0

        lhs = e.T @ c_forward
        rhs = d_transpose.T @ b

        np.testing.assert_allclose(lhs, rhs, rtol=1e-12)


class TestMB02KDErrors:
    """Error handling tests."""

    def test_invalid_ldblk(self):
        """Test invalid LDBLK parameter."""
        tc = np.ones((4, 2), order='F', dtype=float)
        tr = np.ones((2, 2), order='F', dtype=float)
        b = np.ones((4, 1), order='F', dtype=float)

        c, info = mb02kd('X', 'N', 2, 2, 2, 2, 1, 1.0, 0.0, tc, tr, b)

        assert info == -1

    def test_invalid_trans(self):
        """Test invalid TRANS parameter."""
        tc = np.ones((4, 2), order='F', dtype=float)
        tr = np.ones((2, 2), order='F', dtype=float)
        b = np.ones((4, 1), order='F', dtype=float)

        c, info = mb02kd('C', 'X', 2, 2, 2, 2, 1, 1.0, 0.0, tc, tr, b)

        assert info == -2

    def test_negative_k(self):
        """Test negative K parameter."""
        tc = np.ones((4, 2), order='F', dtype=float)
        tr = np.ones((2, 2), order='F', dtype=float)
        b = np.ones((4, 1), order='F', dtype=float)

        c, info = mb02kd('C', 'N', -1, 2, 2, 2, 1, 1.0, 0.0, tc, tr, b)

        assert info == -3
