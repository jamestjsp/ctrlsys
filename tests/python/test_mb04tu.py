"""
Tests for MB04TU: Givens transformation with interchange.

MB04TU applies a row-permuted Givens transformation:
    |X_new|   | 0   1 |   | C   S |   |X|     | C*Y - S*X |
    |     | = |       | x |       | x | |  =  |           |
    |Y_new|   | 1   0 |   |-S   C |   |Y|     | C*X + S*Y |

This is NOT standard DROT which computes:
    X_new = C*X + S*Y
    Y_new = -S*X + C*Y
"""

import numpy as np
import pytest
from ctrlsys import mb04tu


class TestMB04TUBasic:
    """Basic functionality tests."""

    def test_unit_increment_identity_rotation(self):
        """
        Test with C=1, S=0 (identity rotation) which just swaps X and Y.

        X_new = C*Y - S*X = 1*Y - 0*X = Y
        Y_new = C*X + S*Y = 1*X + 0*Y = X
        """
        x = np.array([1.0, 2.0, 3.0], dtype=float, order='F')
        y = np.array([4.0, 5.0, 6.0], dtype=float, order='F')
        c = 1.0
        s = 0.0

        x_orig = x.copy()
        y_orig = y.copy()

        x_out, y_out = mb04tu(x, y, c, s)

        np.testing.assert_allclose(x_out, y_orig, rtol=1e-14)
        np.testing.assert_allclose(y_out, x_orig, rtol=1e-14)

    def test_unit_increment_45_degree_rotation(self):
        """
        Test with C=S=1/sqrt(2) (45 degree rotation with swap).

        X_new = C*Y - S*X = (Y - X)/sqrt(2)
        Y_new = C*X + S*Y = (X + Y)/sqrt(2)
        """
        x = np.array([1.0, 0.0], dtype=float, order='F')
        y = np.array([0.0, 1.0], dtype=float, order='F')
        c = 1.0 / np.sqrt(2.0)
        s = 1.0 / np.sqrt(2.0)

        x_orig = x.copy()
        y_orig = y.copy()

        x_out, y_out = mb04tu(x, y, c, s)

        x_expected = c * y_orig - s * x_orig
        y_expected = c * x_orig + s * y_orig

        np.testing.assert_allclose(x_out, x_expected, rtol=1e-14)
        np.testing.assert_allclose(y_out, y_expected, rtol=1e-14)

    def test_unit_increment_90_degree_rotation(self):
        """
        Test with C=0, S=1 (90 degree rotation with swap).

        X_new = C*Y - S*X = 0*Y - 1*X = -X
        Y_new = C*X + S*Y = 0*X + 1*Y = Y
        """
        x = np.array([3.0, 4.0, 5.0], dtype=float, order='F')
        y = np.array([6.0, 7.0, 8.0], dtype=float, order='F')
        c = 0.0
        s = 1.0

        x_orig = x.copy()
        y_orig = y.copy()

        x_out, y_out = mb04tu(x, y, c, s)

        np.testing.assert_allclose(x_out, -x_orig, rtol=1e-14)
        np.testing.assert_allclose(y_out, y_orig, rtol=1e-14)


class TestMB04TUIncrements:
    """Tests for various increment values."""

    def test_positive_increments(self):
        """Test with incx=2, incy=3 (non-unit positive increments)."""
        x = np.array([1.0, 0.0, 2.0, 0.0, 3.0], dtype=float, order='F')
        y = np.array([4.0, 0.0, 0.0, 5.0, 0.0, 0.0, 6.0], dtype=float, order='F')
        n = 3
        incx = 2
        incy = 3
        c = 0.6
        s = 0.8

        x_vals = np.array([1.0, 2.0, 3.0])
        y_vals = np.array([4.0, 5.0, 6.0])

        x_expected_vals = c * y_vals - s * x_vals
        y_expected_vals = c * x_vals + s * y_vals

        x_out, y_out = mb04tu(x, y, c, s, n=n, incx=incx, incy=incy)

        np.testing.assert_allclose(x_out[::incx][:n], x_expected_vals, rtol=1e-14)
        np.testing.assert_allclose(y_out[::incy][:n], y_expected_vals, rtol=1e-14)

    def test_negative_increment_x(self):
        """
        Test with negative incx.

        When incx < 0, start from index (-n+1)*incx (0-based).
        For n=3, incx=-1: start at 2, step by -1, so accesses indices 2,1,0.

        Pairs: (x[2], y[0]), (x[1], y[1]), (x[0], y[2])
        """
        n = 3
        incx = -1
        incy = 1
        c = 0.6
        s = 0.8

        x = np.array([1.0, 2.0, 3.0], dtype=float, order='F')
        y = np.array([4.0, 5.0, 6.0], dtype=float, order='F')
        x_orig = x.copy()
        y_orig = y.copy()

        x_out, y_out = mb04tu(x.copy(), y.copy(), c, s, n=n, incx=incx, incy=incy)

        x_expected = x_orig.copy()
        y_expected = y_orig.copy()
        ix = 2
        iy = 0
        for _ in range(n):
            temp = c * y_orig[iy] - s * x_orig[ix]
            y_expected[iy] = c * x_orig[ix] + s * y_orig[iy]
            x_expected[ix] = temp
            ix += incx
            iy += incy

        np.testing.assert_allclose(x_out, x_expected, rtol=1e-14)
        np.testing.assert_allclose(y_out, y_expected, rtol=1e-14)


class TestMB04TUMathProperties:
    """Mathematical property validation tests."""

    def test_orthogonality_preservation(self):
        """
        Verify that transformation preserves vector norms when C^2 + S^2 = 1.

        For Givens rotation: ||x||^2 + ||y||^2 is preserved.
        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n = 10
        x = np.random.randn(n).astype(float, order='F')
        y = np.random.randn(n).astype(float, order='F')

        theta = np.pi / 6
        c = np.cos(theta)
        s = np.sin(theta)

        norm_before = np.linalg.norm(x)**2 + np.linalg.norm(y)**2

        x_out, y_out = mb04tu(x, y, c, s)

        norm_after = np.linalg.norm(x_out)**2 + np.linalg.norm(y_out)**2

        np.testing.assert_allclose(norm_after, norm_before, rtol=1e-14)

    def test_involution_property(self):
        """
        Verify applying transformation twice returns to original (up to sign).

        The matrix M = P @ G where P = [[0,1],[1,0]] and G = [[C,S],[-S,C]].
        M^2 = P@G@P@G. This tests consistency of the transformation.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n = 5
        x = np.random.randn(n).astype(float, order='F')
        y = np.random.randn(n).astype(float, order='F')

        c = 0.6
        s = 0.8

        x1, y1 = mb04tu(x.copy(), y.copy(), c, s)
        x2, y2 = mb04tu(x1, y1, c, s)

        np.testing.assert_allclose(
            np.vstack([x2, y2]),
            np.vstack([x, y]),
            rtol=1e-14
        )


class TestMB04TUEdgeCases:
    """Edge case tests."""

    def test_n_zero(self):
        """Test with n=0 returns unchanged arrays."""
        x = np.array([1.0, 2.0], dtype=float, order='F')
        y = np.array([3.0, 4.0], dtype=float, order='F')
        c = 0.6
        s = 0.8

        x_out, y_out = mb04tu(x, y, c, s, n=0)

        np.testing.assert_array_equal(x_out, x)
        np.testing.assert_array_equal(y_out, y)

    def test_n_one(self):
        """Test with single element vectors."""
        x = np.array([2.0], dtype=float, order='F')
        y = np.array([3.0], dtype=float, order='F')
        c = 0.6
        s = 0.8

        x_out, y_out = mb04tu(x, y, c, s)

        np.testing.assert_allclose(x_out[0], c * 3.0 - s * 2.0, rtol=1e-14)
        np.testing.assert_allclose(y_out[0], c * 2.0 + s * 3.0, rtol=1e-14)

    def test_large_vectors(self):
        """
        Test with larger vectors to ensure loop unrolling is correct.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n = 1000
        x = np.random.randn(n).astype(float, order='F')
        y = np.random.randn(n).astype(float, order='F')
        c = 0.8
        s = 0.6

        x_expected = c * y - s * x
        y_expected = c * x + s * y

        x_out, y_out = mb04tu(x.copy(), y.copy(), c, s)

        np.testing.assert_allclose(x_out, x_expected, rtol=5e-13)
        np.testing.assert_allclose(y_out, y_expected, rtol=5e-13)
