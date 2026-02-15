"""Tests for MA02BD: Reversing order of rows and/or columns of a matrix."""

import numpy as np
import pytest
from ctrlsys import ma02bd


class TestMA02BDLeft:
    """Test SIDE='L' - reverse row order (P*A)."""

    def test_left_4x3_matrix(self):
        """Reverse rows of 4x3 matrix. Random seed: 42."""
        np.random.seed(42)
        a = np.random.randn(4, 3).astype(float, order='F')
        a_original = a.copy()

        ma02bd('L', a)

        expected = a_original[::-1, :].copy(order='F')
        np.testing.assert_allclose(a, expected, rtol=1e-14)

    def test_left_5x4_odd_rows(self):
        """Reverse rows of 5x4 matrix (odd row count). Random seed: 123."""
        np.random.seed(123)
        a = np.random.randn(5, 4).astype(float, order='F')
        a_original = a.copy()

        ma02bd('L', a)

        expected = a_original[::-1, :].copy(order='F')
        np.testing.assert_allclose(a, expected, rtol=1e-14)

    def test_left_single_row(self):
        """Single row matrix - no change."""
        a = np.array([[1.0, 2.0, 3.0]], order='F')
        a_original = a.copy()

        ma02bd('L', a)

        np.testing.assert_allclose(a, a_original, rtol=1e-14)

    def test_left_involution(self):
        """Mathematical property: (P*P*A) = A (involution). Random seed: 456."""
        np.random.seed(456)
        a = np.random.randn(6, 4).astype(float, order='F')
        a_original = a.copy()

        ma02bd('L', a)
        ma02bd('L', a)

        np.testing.assert_allclose(a, a_original, rtol=1e-14)


class TestMA02BDRight:
    """Test SIDE='R' - reverse column order (A*P)."""

    def test_right_3x4_matrix(self):
        """Reverse columns of 3x4 matrix. Random seed: 789."""
        np.random.seed(789)
        a = np.random.randn(3, 4).astype(float, order='F')
        a_original = a.copy()

        ma02bd('R', a)

        expected = a_original[:, ::-1].copy(order='F')
        np.testing.assert_allclose(a, expected, rtol=1e-14)

    def test_right_4x5_odd_cols(self):
        """Reverse columns of 4x5 matrix (odd col count). Random seed: 111."""
        np.random.seed(111)
        a = np.random.randn(4, 5).astype(float, order='F')
        a_original = a.copy()

        ma02bd('R', a)

        expected = a_original[:, ::-1].copy(order='F')
        np.testing.assert_allclose(a, expected, rtol=1e-14)

    def test_right_single_col(self):
        """Single column matrix - no change."""
        a = np.array([[1.0], [2.0], [3.0]], order='F')
        a_original = a.copy()

        ma02bd('R', a)

        np.testing.assert_allclose(a, a_original, rtol=1e-14)

    def test_right_involution(self):
        """Mathematical property: (A*P*P) = A (involution). Random seed: 222."""
        np.random.seed(222)
        a = np.random.randn(4, 6).astype(float, order='F')
        a_original = a.copy()

        ma02bd('R', a)
        ma02bd('R', a)

        np.testing.assert_allclose(a, a_original, rtol=1e-14)


class TestMA02BDBoth:
    """Test SIDE='B' - reverse both rows and columns (P*A*P)."""

    def test_both_4x4_square(self):
        """Reverse both rows and columns of 4x4 matrix. Random seed: 333."""
        np.random.seed(333)
        a = np.random.randn(4, 4).astype(float, order='F')
        a_original = a.copy()

        ma02bd('B', a)

        expected = a_original[::-1, ::-1].copy(order='F')
        np.testing.assert_allclose(a, expected, rtol=1e-14)

    def test_both_3x5_rectangular(self):
        """Reverse both for 3x5 matrix. Random seed: 444."""
        np.random.seed(444)
        a = np.random.randn(3, 5).astype(float, order='F')
        a_original = a.copy()

        ma02bd('B', a)

        expected = a_original[::-1, ::-1].copy(order='F')
        np.testing.assert_allclose(a, expected, rtol=1e-14)

    def test_both_involution(self):
        """Mathematical property: (P*P*A*P*P) = A (involution). Random seed: 555."""
        np.random.seed(555)
        a = np.random.randn(5, 6).astype(float, order='F')
        a_original = a.copy()

        ma02bd('B', a)
        ma02bd('B', a)

        np.testing.assert_allclose(a, a_original, rtol=1e-14)

    def test_both_equals_left_then_right(self):
        """Property: P*A*P = (P*(A*P)) = ((P*A)*P). Random seed: 666."""
        np.random.seed(666)
        a1 = np.random.randn(4, 5).astype(float, order='F')
        a2 = a1.copy()
        a3 = a1.copy()

        ma02bd('B', a1)

        ma02bd('R', a2)
        ma02bd('L', a2)

        ma02bd('L', a3)
        ma02bd('R', a3)

        np.testing.assert_allclose(a1, a2, rtol=1e-14)
        np.testing.assert_allclose(a1, a3, rtol=1e-14)


class TestMA02BDEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_matrix(self):
        """Empty matrix (0x0) - should handle gracefully."""
        a = np.array([], dtype=float, order='F').reshape(0, 0)
        ma02bd('B', a)
        assert a.shape == (0, 0)

    def test_1x1_matrix(self):
        """1x1 matrix - no change."""
        a = np.array([[5.0]], order='F')
        a_original = a.copy()

        ma02bd('B', a)

        np.testing.assert_allclose(a, a_original, rtol=1e-14)

    def test_lowercase_side(self):
        """Lowercase 'l' should work like 'L'."""
        np.random.seed(777)
        a1 = np.random.randn(3, 4).astype(float, order='F')
        a2 = a1.copy()

        ma02bd('L', a1)
        ma02bd('l', a2)

        np.testing.assert_allclose(a1, a2, rtol=1e-14)

    def test_2x2_matrix_left(self):
        """2x2 matrix left reversal - explicit values."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], order='F')

        ma02bd('L', a)

        expected = np.array([[3.0, 4.0], [1.0, 2.0]], order='F')
        np.testing.assert_allclose(a, expected, rtol=1e-14)

    def test_2x2_matrix_right(self):
        """2x2 matrix right reversal - explicit values."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], order='F')

        ma02bd('R', a)

        expected = np.array([[2.0, 1.0], [4.0, 3.0]], order='F')
        np.testing.assert_allclose(a, expected, rtol=1e-14)

    def test_2x2_matrix_both(self):
        """2x2 matrix both reversal - explicit values."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], order='F')

        ma02bd('B', a)

        expected = np.array([[4.0, 3.0], [2.0, 1.0]], order='F')
        np.testing.assert_allclose(a, expected, rtol=1e-14)
