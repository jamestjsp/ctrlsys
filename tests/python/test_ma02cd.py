"""Tests for MA02CD - Pertranspose of central band of a square matrix.

The pertranspose operation reverses elements along each antidiagonal within
the specified band (KL subdiagonals, main diagonal, KU superdiagonals).
Mathematically equivalent to P*B'*P where B is the band and P is the
permutation matrix with ones on the secondary diagonal.
"""

import numpy as np
import pytest
from ctrlsys import ma02cd


def _pertranspose_band_reference(a, kl, ku):
    """Reference implementation of band pertranspose.

    For element at (i,j) within the band, it swaps with element at (n-1-j, n-1-i).
    This is equivalent to P*B'*P where P reverses row/column order.
    """
    n = a.shape[0]
    result = a.copy()

    for i in range(n):
        for j in range(n):
            if j - i <= ku and i - j <= kl:
                ni, nj = n - 1 - j, n - 1 - i
                if nj - ni <= ku and ni - nj <= kl:
                    if i < n - 1 - j or (i == n - 1 - j and j < n - 1 - i):
                        result[i, j], result[ni, nj] = a[ni, nj], a[i, j]
    return result


class TestMA02CDBasic:
    """Basic functionality tests."""

    def test_full_matrix_4x4(self):
        """Pertranspose full 4x4 matrix (KL=3, KU=3).

        For full matrix pertranspose: element (i,j) swaps with (n-1-j, n-1-i).
        """
        a = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ], order='F', dtype=float)

        expected = np.array([
            [16.0, 12.0, 8.0, 4.0],
            [15.0, 11.0, 7.0, 3.0],
            [14.0, 10.0, 6.0, 2.0],
            [13.0, 9.0, 5.0, 1.0]
        ], order='F', dtype=float)

        ma02cd(a, 3, 3)

        np.testing.assert_allclose(a, expected, rtol=1e-14)

    def test_diagonal_only(self):
        """Pertranspose only the diagonal (KL=0, KU=0).

        Diagonal elements swap: a[i,i] <-> a[n-1-i, n-1-i].
        """
        a = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ], order='F', dtype=float)

        expected = np.array([
            [16.0, 2.0, 3.0, 4.0],
            [5.0, 11.0, 7.0, 8.0],
            [9.0, 10.0, 6.0, 12.0],
            [13.0, 14.0, 15.0, 1.0]
        ], order='F', dtype=float)

        ma02cd(a, 0, 0)

        np.testing.assert_allclose(a, expected, rtol=1e-14)

    def test_tridiagonal(self):
        """Pertranspose tridiagonal band (KL=1, KU=1).

        Only main diagonal and one sub/super diagonal are affected.
        """
        a = np.array([
            [1.0, 2.0, 0.0, 0.0],
            [5.0, 6.0, 7.0, 0.0],
            [0.0, 10.0, 11.0, 12.0],
            [0.0, 0.0, 15.0, 16.0]
        ], order='F', dtype=float)

        expected = np.array([
            [16.0, 12.0, 0.0, 0.0],
            [15.0, 11.0, 7.0, 0.0],
            [0.0, 10.0, 6.0, 2.0],
            [0.0, 0.0, 5.0, 1.0]
        ], order='F', dtype=float)

        ma02cd(a, 1, 1)

        np.testing.assert_allclose(a, expected, rtol=1e-14)


class TestMA02CDMathProperties:
    """Mathematical property tests."""

    def test_involution_full_matrix(self):
        """Pertranspose is an involution: applying twice returns original.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n = 5
        a_orig = np.random.randn(n, n).astype(float, order='F')
        a = a_orig.copy()

        ma02cd(a, n-1, n-1)
        ma02cd(a, n-1, n-1)

        np.testing.assert_allclose(a, a_orig, rtol=1e-14)

    def test_involution_banded(self):
        """Pertranspose is an involution for banded matrices.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n = 6
        kl, ku = 2, 1
        a_orig = np.random.randn(n, n).astype(float, order='F')
        a = a_orig.copy()

        ma02cd(a, kl, ku)
        ma02cd(a, kl, ku)

        np.testing.assert_allclose(a, a_orig, rtol=1e-14)

    def test_trace_preserved_symmetric_band(self):
        """For symmetric band, trace is preserved under pertranspose.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n = 4
        a = np.random.randn(n, n).astype(float, order='F')
        trace_before = np.trace(a)

        ma02cd(a, n-1, n-1)
        trace_after = np.trace(a)

        np.testing.assert_allclose(trace_before, trace_after, rtol=1e-14)

    def test_equivalence_to_pbp(self):
        """Pertranspose equals P*A'*P where P is the exchange matrix.

        The exchange matrix P has ones on the anti-diagonal.
        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n = 5
        a = np.random.randn(n, n).astype(float, order='F')
        a_copy = a.copy()

        p = np.eye(n, dtype=float, order='F')[:, ::-1]
        expected = p @ a.T @ p

        ma02cd(a, n-1, n-1)

        np.testing.assert_allclose(a, expected, rtol=1e-14)


class TestMA02CDEdgeCases:
    """Edge case tests."""

    def test_n_equals_1(self):
        """N=1 matrix should be unchanged (quick return)."""
        a = np.array([[5.0]], order='F', dtype=float)
        a_orig = a.copy()

        ma02cd(a, 0, 0)

        np.testing.assert_allclose(a, a_orig, rtol=1e-14)

    def test_n_equals_2(self):
        """2x2 matrix pertranspose.

        For N=2, min(KL,N-2)=min(1,0)=0 and min(KU,N-2)=0, so only diagonal
        is pertransposed (elements (0,0) and (1,1) are swapped).
        """
        a = np.array([
            [1.0, 2.0],
            [3.0, 4.0]
        ], order='F', dtype=float)

        expected = np.array([
            [4.0, 2.0],
            [3.0, 1.0]
        ], order='F', dtype=float)

        ma02cd(a, 1, 1)

        np.testing.assert_allclose(a, expected, rtol=1e-14)

    def test_odd_dimension(self):
        """Odd-dimension matrix (center element stays in place for diagonal).

        For 3x3 with full pertranspose, center element a[1,1] stays fixed.
        """
        a = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ], order='F', dtype=float)

        expected = np.array([
            [9.0, 6.0, 3.0],
            [8.0, 5.0, 2.0],
            [7.0, 4.0, 1.0]
        ], order='F', dtype=float)

        ma02cd(a, 2, 2)

        np.testing.assert_allclose(a, expected, rtol=1e-14)

    def test_asymmetric_band(self):
        """Asymmetric band: more subdiagonals than superdiagonals.

        KL=2, KU=0: only subdiagonals and diagonal are pertransposed.
        """
        a = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ], order='F', dtype=float)

        expected = np.array([
            [16.0, 2.0, 3.0, 4.0],
            [15.0, 11.0, 7.0, 8.0],
            [14.0, 10.0, 6.0, 12.0],
            [13.0, 9.0, 5.0, 1.0]
        ], order='F', dtype=float)

        ma02cd(a, 2, 0)

        np.testing.assert_allclose(a, expected, rtol=1e-14)

    def test_superdiagonal_only(self):
        """Only superdiagonals pertransposed (KL=0, KU=2)."""
        a = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ], order='F', dtype=float)

        expected = np.array([
            [16.0, 12.0, 8.0, 4.0],
            [5.0, 11.0, 7.0, 3.0],
            [9.0, 10.0, 6.0, 2.0],
            [13.0, 14.0, 15.0, 1.0]
        ], order='F', dtype=float)

        ma02cd(a, 0, 2)

        np.testing.assert_allclose(a, expected, rtol=1e-14)


class TestMA02CDLarger:
    """Larger matrix tests."""

    def test_6x6_banded(self):
        """6x6 matrix with KL=1, KU=2.

        Random seed: 999 (for reproducibility)
        """
        np.random.seed(999)
        n = 6
        kl, ku = 1, 2
        a = np.random.randn(n, n).astype(float, order='F')

        p = np.eye(n, dtype=float, order='F')[:, ::-1]

        b = np.zeros_like(a)
        for i in range(n):
            for j in range(n):
                if j - i <= ku and i - j <= kl:
                    b[i, j] = a[i, j]

        expected = a.copy()
        band_pertrans = p @ b.T @ p
        for i in range(n):
            for j in range(n):
                if j - i <= ku and i - j <= kl:
                    expected[i, j] = band_pertrans[i, j]

        ma02cd(a, kl, ku)

        np.testing.assert_allclose(a, expected, rtol=1e-14)
