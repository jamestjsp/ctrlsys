"""
Tests for MB01UX: Quasi-triangular matrix-matrix product A := alpha*op(T)*A or A := alpha*A*op(T)
"""
import numpy as np
import pytest
from ctrlsys import mb01ux


class TestMB01UXBasic:
    """Basic functionality tests."""

    def test_left_upper_notrans_3x2(self):
        """
        Test SIDE='L', UPLO='U', TRANS='N': A := alpha*T*A with upper quasi-triangular T.

        Random seed: 42 (for reproducibility)
        """
        m, n = 3, 2
        alpha = 2.0

        t = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [0.0, 7.0, 8.0]
        ], order='F', dtype=float)

        a = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ], order='F', dtype=float)

        t_full = np.triu(t, -1)

        expected = alpha * (t_full @ a)

        result, info = mb01ux('L', 'U', 'N', m, n, alpha, t.copy(), a.copy())

        assert info == 0
        np.testing.assert_allclose(result, expected, rtol=1e-13)

    def test_left_upper_trans_3x2(self):
        """
        Test SIDE='L', UPLO='U', TRANS='T': A := alpha*T'*A with upper quasi-triangular T.

        Random seed: 43 (for reproducibility)
        """
        m, n = 3, 2
        alpha = 1.5

        t = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [0.0, 7.0, 8.0]
        ], order='F', dtype=float)

        a = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ], order='F', dtype=float)

        t_full = np.triu(t, -1)

        expected = alpha * (t_full.T @ a)

        result, info = mb01ux('L', 'U', 'T', m, n, alpha, t.copy(), a.copy())

        assert info == 0
        np.testing.assert_allclose(result, expected, rtol=1e-13)

    def test_right_upper_notrans_2x3(self):
        """
        Test SIDE='R', UPLO='U', TRANS='N': A := alpha*A*T with upper quasi-triangular T.

        Random seed: 44 (for reproducibility)
        """
        m, n = 2, 3
        alpha = 0.5

        t = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [0.0, 7.0, 8.0]
        ], order='F', dtype=float)

        a = np.array([
            [1.0, 3.0, 5.0],
            [2.0, 4.0, 6.0]
        ], order='F', dtype=float)

        t_full = np.triu(t, -1)

        expected = alpha * (a @ t_full)

        result, info = mb01ux('R', 'U', 'N', m, n, alpha, t.copy(), a.copy())

        assert info == 0
        np.testing.assert_allclose(result, expected, rtol=1e-13)

    def test_right_upper_trans_2x3(self):
        """
        Test SIDE='R', UPLO='U', TRANS='T': A := alpha*A*T' with upper quasi-triangular T.

        Random seed: 45 (for reproducibility)
        """
        m, n = 2, 3
        alpha = 3.0

        t = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [0.0, 7.0, 8.0]
        ], order='F', dtype=float)

        a = np.array([
            [1.0, 3.0, 5.0],
            [2.0, 4.0, 6.0]
        ], order='F', dtype=float)

        t_full = np.triu(t, -1)

        expected = alpha * (a @ t_full.T)

        result, info = mb01ux('R', 'U', 'T', m, n, alpha, t.copy(), a.copy())

        assert info == 0
        np.testing.assert_allclose(result, expected, rtol=1e-13)

    def test_left_lower_notrans_3x2(self):
        """
        Test SIDE='L', UPLO='L', TRANS='N': A := alpha*T*A with lower quasi-triangular T.

        Random seed: 46 (for reproducibility)
        """
        m, n = 3, 2
        alpha = 2.0

        t = np.array([
            [1.0, 4.0, 0.0],
            [2.0, 5.0, 7.0],
            [3.0, 6.0, 8.0]
        ], order='F', dtype=float)

        a = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ], order='F', dtype=float)

        t_full = np.tril(t, 1)

        expected = alpha * (t_full @ a)

        result, info = mb01ux('L', 'L', 'N', m, n, alpha, t.copy(), a.copy())

        assert info == 0
        np.testing.assert_allclose(result, expected, rtol=1e-13)


class TestMB01UXEdgeCases:
    """Edge case tests."""

    def test_alpha_zero(self):
        """When alpha=0, result should be zero matrix."""
        m, n = 3, 2
        alpha = 0.0

        t = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [0.0, 7.0, 8.0]
        ], order='F', dtype=float)

        a = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ], order='F', dtype=float)

        result, info = mb01ux('L', 'U', 'N', m, n, alpha, t.copy(), a.copy())

        assert info == 0
        np.testing.assert_allclose(result, np.zeros((m, n)), rtol=1e-14)

    def test_empty_matrix(self):
        """Test with m=0 or n=0."""
        t = np.array([[1.0]], order='F', dtype=float)
        a_empty = np.zeros((0, 2), order='F', dtype=float)

        result, info = mb01ux('L', 'U', 'N', 0, 2, 1.0, t, a_empty)

        assert info == 0
        assert result.shape[0] == 0 or result.size == 0

    def test_identity_quasi_triangular(self):
        """
        When T is identity, A := alpha*I*A = alpha*A.

        Random seed: 100 (for reproducibility)
        """
        np.random.seed(100)
        m, n = 4, 3
        alpha = 2.5

        t = np.eye(m, order='F', dtype=float)
        a = np.random.randn(m, n).astype(float, order='F')
        a_copy = a.copy()

        expected = alpha * a_copy

        result, info = mb01ux('L', 'U', 'N', m, n, alpha, t.copy(), a.copy())

        assert info == 0
        np.testing.assert_allclose(result, expected, rtol=1e-14)


class TestMB01UXMathProperties:
    """Mathematical property tests."""

    def test_scaling_consistency_upper(self):
        """
        Test alpha*(T*A) = T*(alpha*A) for numerical consistency.

        Random seed: 103 (for reproducibility)
        """
        np.random.seed(103)
        m, n = 4, 3
        alpha = 2.5

        t = np.triu(np.random.randn(m, m), -1).astype(float, order='F')
        a = np.random.randn(m, n).astype(float, order='F')

        result1, info1 = mb01ux('L', 'U', 'N', m, n, alpha, t.copy(), a.copy())
        assert info1 == 0

        result2, info2 = mb01ux('L', 'U', 'N', m, n, 1.0, t.copy(), (alpha * a).copy())
        assert info2 == 0

        np.testing.assert_allclose(result1, result2, rtol=1e-13)

    def test_transpose_property(self):
        """
        Test (T'*A)' = A'*T.

        Random seed: 102 (for reproducibility)
        """
        np.random.seed(102)
        m, n = 4, 3
        alpha = 1.0

        t = np.triu(np.random.randn(m, m), -1).astype(float, order='F')
        a = np.random.randn(m, n).astype(float, order='F')

        result, info = mb01ux('L', 'U', 'T', m, n, alpha, t.copy(), a.copy())
        assert info == 0

        t_full = np.triu(t, -1)
        expected = t_full.T @ a

        np.testing.assert_allclose(result, expected, rtol=1e-13)

    def test_right_side_matrix_product(self):
        """
        Test A*T equals expected for right-side multiplication.

        Random seed: 104 (for reproducibility)
        """
        np.random.seed(104)
        m, n = 3, 4
        alpha = 1.0

        t = np.triu(np.random.randn(n, n), -1).astype(float, order='F')
        a = np.random.randn(m, n).astype(float, order='F')

        result, info = mb01ux('R', 'U', 'N', m, n, alpha, t.copy(), a.copy())
        assert info == 0

        t_full = np.triu(t, -1)
        expected = a @ t_full

        np.testing.assert_allclose(result, expected, rtol=1e-13)


class TestMB01UXErrors:
    """Error handling tests."""

    def test_invalid_side(self):
        """Test invalid SIDE parameter."""
        t = np.eye(3, order='F', dtype=float)
        a = np.ones((3, 2), order='F', dtype=float)

        _, info = mb01ux('X', 'U', 'N', 3, 2, 1.0, t, a)

        assert info == -1

    def test_invalid_uplo(self):
        """Test invalid UPLO parameter."""
        t = np.eye(3, order='F', dtype=float)
        a = np.ones((3, 2), order='F', dtype=float)

        _, info = mb01ux('L', 'X', 'N', 3, 2, 1.0, t, a)

        assert info == -2

    def test_invalid_trans(self):
        """Test invalid TRANS parameter."""
        t = np.eye(3, order='F', dtype=float)
        a = np.ones((3, 2), order='F', dtype=float)

        _, info = mb01ux('L', 'U', 'X', 3, 2, 1.0, t, a)

        assert info == -3

    def test_negative_m(self):
        """Test negative M parameter."""
        t = np.eye(3, order='F', dtype=float)
        a = np.ones((3, 2), order='F', dtype=float)

        _, info = mb01ux('L', 'U', 'N', -1, 2, 1.0, t, a)

        assert info == -4

    def test_negative_n(self):
        """Test negative N parameter."""
        t = np.eye(3, order='F', dtype=float)
        a = np.ones((3, 2), order='F', dtype=float)

        _, info = mb01ux('L', 'U', 'N', 3, -1, 1.0, t, a)

        assert info == -5


class TestMB01UXLargerMatrices:
    """Tests with larger matrices to exercise BLAS3 path."""

    def test_large_left_upper_notrans(self):
        """
        Test larger matrix to exercise BLAS3 DTRMM path.

        Random seed: 200 (for reproducibility)
        """
        np.random.seed(200)
        m, n = 10, 8
        alpha = 1.5

        t = np.triu(np.random.randn(m, m), -1).astype(float, order='F')
        a = np.random.randn(m, n).astype(float, order='F')

        t_full = np.triu(t, -1)
        expected = alpha * (t_full @ a)

        result, info = mb01ux('L', 'U', 'N', m, n, alpha, t.copy(), a.copy())

        assert info == 0
        np.testing.assert_allclose(result, expected, rtol=1e-13)

    def test_large_right_upper_trans(self):
        """
        Test larger matrix for SIDE='R', TRANS='T'.

        Random seed: 201 (for reproducibility)
        """
        np.random.seed(201)
        m, n = 8, 10
        alpha = 0.7

        t = np.triu(np.random.randn(n, n), -1).astype(float, order='F')
        a = np.random.randn(m, n).astype(float, order='F')

        t_full = np.triu(t, -1)
        expected = alpha * (a @ t_full.T)

        result, info = mb01ux('R', 'U', 'T', m, n, alpha, t.copy(), a.copy())

        assert info == 0
        np.testing.assert_allclose(result, expected, rtol=1e-13)

    def test_large_left_lower_notrans(self):
        """
        Test larger matrix with lower quasi-triangular.

        Random seed: 202 (for reproducibility)
        """
        np.random.seed(202)
        m, n = 10, 8
        alpha = 1.2

        t = np.tril(np.random.randn(m, m), 1).astype(float, order='F')
        a = np.random.randn(m, n).astype(float, order='F')

        t_full = np.tril(t, 1)
        expected = alpha * (t_full @ a)

        result, info = mb01ux('L', 'L', 'N', m, n, alpha, t.copy(), a.copy())

        assert info == 0
        np.testing.assert_allclose(result, expected, rtol=1e-13)
