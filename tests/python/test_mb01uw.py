"""
Tests for MB01UW: Hessenberg matrix-matrix product A := alpha*op(H)*A or A := alpha*A*op(H)
"""
import numpy as np
import pytest
from ctrlsys import mb01uw


class TestMB01UWBasic:
    """Basic functionality tests."""

    def test_left_notrans_3x2(self):
        """
        Test SIDE='L', TRANS='N': A := alpha*H*A

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        m, n = 3, 2
        alpha = 2.0

        h = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [0.0, 7.0, 8.0]
        ], order='F', dtype=float)

        a = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ], order='F', dtype=float)

        h_full = h.copy()
        h_full[2, 0] = 0.0

        expected = alpha * (h_full @ a)

        result, info = mb01uw('L', 'N', m, n, alpha, h.copy(), a.copy())

        assert info == 0
        np.testing.assert_allclose(result, expected, rtol=1e-14)

    def test_left_trans_3x2(self):
        """
        Test SIDE='L', TRANS='T': A := alpha*H'*A

        Random seed: 43 (for reproducibility)
        """
        np.random.seed(43)
        m, n = 3, 2
        alpha = 1.5

        h = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [0.0, 7.0, 8.0]
        ], order='F', dtype=float)

        a = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ], order='F', dtype=float)

        h_full = h.copy()
        h_full[2, 0] = 0.0

        expected = alpha * (h_full.T @ a)

        result, info = mb01uw('L', 'T', m, n, alpha, h.copy(), a.copy())

        assert info == 0
        np.testing.assert_allclose(result, expected, rtol=1e-14)

    def test_right_notrans_2x3(self):
        """
        Test SIDE='R', TRANS='N': A := alpha*A*H

        Random seed: 44 (for reproducibility)
        """
        np.random.seed(44)
        m, n = 2, 3
        alpha = 0.5

        h = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [0.0, 7.0, 8.0]
        ], order='F', dtype=float)

        a = np.array([
            [1.0, 3.0, 5.0],
            [2.0, 4.0, 6.0]
        ], order='F', dtype=float)

        h_full = h.copy()
        h_full[2, 0] = 0.0

        expected = alpha * (a @ h_full)

        result, info = mb01uw('R', 'N', m, n, alpha, h.copy(), a.copy())

        assert info == 0
        np.testing.assert_allclose(result, expected, rtol=1e-14)

    def test_right_trans_2x3(self):
        """
        Test SIDE='R', TRANS='T': A := alpha*A*H'

        Random seed: 45 (for reproducibility)
        """
        np.random.seed(45)
        m, n = 2, 3
        alpha = 3.0

        h = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [0.0, 7.0, 8.0]
        ], order='F', dtype=float)

        a = np.array([
            [1.0, 3.0, 5.0],
            [2.0, 4.0, 6.0]
        ], order='F', dtype=float)

        h_full = h.copy()
        h_full[2, 0] = 0.0

        expected = alpha * (a @ h_full.T)

        result, info = mb01uw('R', 'T', m, n, alpha, h.copy(), a.copy())

        assert info == 0
        np.testing.assert_allclose(result, expected, rtol=1e-14)


class TestMB01UWEdgeCases:
    """Edge case tests."""

    def test_alpha_zero(self):
        """When alpha=0, result should be zero matrix."""
        m, n = 3, 2
        alpha = 0.0

        h = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [0.0, 7.0, 8.0]
        ], order='F', dtype=float)

        a = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ], order='F', dtype=float)

        result, info = mb01uw('L', 'N', m, n, alpha, h.copy(), a.copy())

        assert info == 0
        np.testing.assert_allclose(result, np.zeros((m, n)), rtol=1e-14)

    def test_1x1_left(self):
        """Test 1x1 Hessenberg matrix (SIDE='L')."""
        m, n = 1, 3
        alpha = 2.0

        h = np.array([[5.0]], order='F', dtype=float)
        a = np.array([[1.0, 2.0, 3.0]], order='F', dtype=float)

        expected = alpha * h[0, 0] * a

        result, info = mb01uw('L', 'N', m, n, alpha, h.copy(), a.copy())

        assert info == 0
        np.testing.assert_allclose(result, expected, rtol=1e-14)

    def test_1x1_right(self):
        """Test 1x1 Hessenberg matrix (SIDE='R')."""
        m, n = 3, 1
        alpha = 2.0

        h = np.array([[5.0]], order='F', dtype=float)
        a = np.array([[1.0], [2.0], [3.0]], order='F', dtype=float)

        expected = alpha * h[0, 0] * a

        result, info = mb01uw('R', 'N', m, n, alpha, h.copy(), a.copy())

        assert info == 0
        np.testing.assert_allclose(result, expected, rtol=1e-14)

    def test_empty_matrix(self):
        """Test with m=0 or n=0."""
        h = np.array([[1.0]], order='F', dtype=float)
        a_empty = np.zeros((0, 2), order='F', dtype=float)

        result, info = mb01uw('L', 'N', 0, 2, 1.0, h, a_empty)

        assert info == 0
        assert result.shape[0] == 0 or result.size == 0


class TestMB01UWMathProperties:
    """Mathematical property tests."""

    def test_identity_hessenberg(self):
        """
        When H is identity, A := alpha*I*A = alpha*A.

        Random seed: 100 (for reproducibility)
        """
        np.random.seed(100)
        m, n = 4, 3
        alpha = 2.5

        h = np.eye(m, order='F', dtype=float)
        a = np.random.randn(m, n).astype(float, order='F')
        a_copy = a.copy()

        expected = alpha * a_copy

        result, info = mb01uw('L', 'N', m, n, alpha, h.copy(), a.copy())

        assert info == 0
        np.testing.assert_allclose(result, expected, rtol=1e-14)

    def test_associativity(self):
        """
        Test (H*A) computed via routine equals manual H@A.

        Random seed: 101 (for reproducibility)
        """
        np.random.seed(101)
        m, n = 4, 3
        alpha = 1.0

        h = np.triu(np.random.randn(m, m), -1).astype(float, order='F')
        a = np.random.randn(m, n).astype(float, order='F')

        expected = h @ a

        result, info = mb01uw('L', 'N', m, n, alpha, h.copy(), a.copy())

        assert info == 0
        np.testing.assert_allclose(result, expected, rtol=1e-13)

    def test_transpose_property(self):
        """
        Test (H'*A)' = A'*H.

        Random seed: 102 (for reproducibility)
        """
        np.random.seed(102)
        m, n = 4, 3
        alpha = 1.0

        h = np.triu(np.random.randn(m, m), -1).astype(float, order='F')
        a = np.random.randn(m, n).astype(float, order='F')

        result, info = mb01uw('L', 'T', m, n, alpha, h.copy(), a.copy())
        assert info == 0

        expected_ht_a = h.T @ a

        np.testing.assert_allclose(result, expected_ht_a, rtol=1e-13)

    def test_scaling_consistency(self):
        """
        Test alpha*(H*A) = H*(alpha*A) for numerical consistency.

        Random seed: 103 (for reproducibility)
        """
        np.random.seed(103)
        m, n = 4, 3
        alpha = 2.5

        h = np.triu(np.random.randn(m, m), -1).astype(float, order='F')
        a = np.random.randn(m, n).astype(float, order='F')

        result1, info1 = mb01uw('L', 'N', m, n, alpha, h.copy(), a.copy())
        assert info1 == 0

        result2, info2 = mb01uw('L', 'N', m, n, 1.0, h.copy(), (alpha * a).copy())
        assert info2 == 0

        np.testing.assert_allclose(result1, result2, rtol=1e-13)


class TestMB01UWErrors:
    """Error handling tests."""

    def test_invalid_side(self):
        """Test invalid SIDE parameter."""
        h = np.eye(3, order='F', dtype=float)
        a = np.ones((3, 2), order='F', dtype=float)

        _, info = mb01uw('X', 'N', 3, 2, 1.0, h, a)

        assert info == -1

    def test_invalid_trans(self):
        """Test invalid TRANS parameter."""
        h = np.eye(3, order='F', dtype=float)
        a = np.ones((3, 2), order='F', dtype=float)

        _, info = mb01uw('L', 'X', 3, 2, 1.0, h, a)

        assert info == -2

    def test_negative_m(self):
        """Test negative M parameter."""
        h = np.eye(3, order='F', dtype=float)
        a = np.ones((3, 2), order='F', dtype=float)

        _, info = mb01uw('L', 'N', -1, 2, 1.0, h, a)

        assert info == -3

    def test_negative_n(self):
        """Test negative N parameter."""
        h = np.eye(3, order='F', dtype=float)
        a = np.ones((3, 2), order='F', dtype=float)

        _, info = mb01uw('L', 'N', 3, -1, 1.0, h, a)

        assert info == -4


class TestMB01UWLargerMatrices:
    """Tests with larger matrices to exercise BLAS3 path."""

    def test_large_left_notrans(self):
        """
        Test larger matrix to exercise BLAS3 DTRMM path.

        Random seed: 200 (for reproducibility)
        """
        np.random.seed(200)
        m, n = 10, 8
        alpha = 1.5

        h = np.triu(np.random.randn(m, m), -1).astype(float, order='F')
        a = np.random.randn(m, n).astype(float, order='F')

        expected = alpha * (h @ a)

        result, info = mb01uw('L', 'N', m, n, alpha, h.copy(), a.copy())

        assert info == 0
        np.testing.assert_allclose(result, expected, rtol=1e-13)

    def test_large_right_trans(self):
        """
        Test larger matrix for SIDE='R', TRANS='T'.

        Random seed: 201 (for reproducibility)
        """
        np.random.seed(201)
        m, n = 8, 10
        alpha = 0.7

        h = np.triu(np.random.randn(n, n), -1).astype(float, order='F')
        a = np.random.randn(m, n).astype(float, order='F')

        expected = alpha * (a @ h.T)

        result, info = mb01uw('R', 'T', m, n, alpha, h.copy(), a.copy())

        assert info == 0
        np.testing.assert_allclose(result, expected, rtol=1e-13)
