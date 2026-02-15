"""
Tests for MA02NZ: Permute two rows and corresponding columns of a
(skew-)symmetric/Hermitian complex matrix.

MA02NZ handles complex matrices with three mode parameters:
- UPLO: 'U' (upper triangle stored) or 'L' (lower triangle stored)
- TRANS: 'T' (transpose) or 'C' (conjugate transpose)
- SKEW: 'N' (symmetric/Hermitian) or 'S' (skew-symmetric/skew-Hermitian)

The routine swaps rows K and L, and corresponding columns K and L,
while maintaining the symmetry structure.

Random seeds used for reproducibility:
- test_hermitian_lower_basic: 42
- test_hermitian_upper_basic: 123
- test_skew_hermitian_lower_basic: 456
- test_involution_property: 789
"""

import numpy as np
import pytest

from ctrlsys import ma02nz


class TestMA02NZHermitianLower:
    """Tests for Hermitian matrix with lower triangle stored (UPLO='L', TRANS='C', SKEW='N')."""

    def test_hermitian_lower_basic_3x3(self):
        """
        Test basic permutation of rows/cols 1 and 3 on a 3x3 Hermitian matrix.

        Lower triangle stored, conjugate transpose, symmetric/Hermitian.
        Permuting rows/cols 1 and 3 (1-based indices K=1, L=3).
        """
        A = np.array([
            [1+0j,  0+0j,  0+0j],
            [2+1j,  3+0j,  0+0j],
            [4+2j,  5+3j,  6+0j]
        ], order='F', dtype=np.complex128)

        expected = np.array([
            [6+0j,  0+0j,  0+0j],
            [5-3j,  3+0j,  0+0j],
            [4-2j,  2-1j,  1+0j]
        ], order='F', dtype=np.complex128)

        ma02nz('L', 'C', 'N', 1, 3, A)

        np.testing.assert_allclose(A, expected, rtol=1e-14)

    def test_hermitian_lower_permute_adjacent(self):
        """
        Test permutation of adjacent rows/cols 2 and 3 on a 4x4 Hermitian matrix.

        K=2, L=3 (1-based).

        For Hermitian matrix, permuting rows/cols K and L is equivalent to P*A*P^T.
        The lower triangle storage after permutation:
        - Elements before col K are swapped between rows K and L (no conjugation)
        - The (L,K) element is conjugated
        - Elements between K and L are conjugated and swapped
        - Elements after row L are swapped (no conjugation)
        """
        A = np.array([
            [1+0j,   0+0j,    0+0j,   0+0j],
            [2+1j,   4+0j,    0+0j,   0+0j],
            [3+2j,   5+3j,    7+0j,   0+0j],
            [6+4j,   8+5j,    9+6j,  10+0j]
        ], order='F', dtype=np.complex128)

        expected = np.array([
            [1+0j,   0+0j,    0+0j,   0+0j],
            [3+2j,   7+0j,    0+0j,   0+0j],
            [2+1j,   5-3j,    4+0j,   0+0j],
            [6+4j,   9+6j,    8+5j,  10+0j]
        ], order='F', dtype=np.complex128)

        ma02nz('L', 'C', 'N', 2, 3, A)

        np.testing.assert_allclose(A, expected, rtol=1e-14)


class TestMA02NZHermitianUpper:
    """Tests for Hermitian matrix with upper triangle stored (UPLO='U', TRANS='C', SKEW='N')."""

    def test_hermitian_upper_basic_3x3(self):
        """
        Test basic permutation of rows/cols 1 and 3 on a 3x3 Hermitian matrix.

        Upper triangle stored, conjugate transpose, symmetric/Hermitian.
        """
        A = np.array([
            [1+0j,  2+1j,  4+2j],
            [0+0j,  3+0j,  5+3j],
            [0+0j,  0+0j,  6+0j]
        ], order='F', dtype=np.complex128)

        expected = np.array([
            [6+0j,  5-3j,  4-2j],
            [0+0j,  3+0j,  2-1j],
            [0+0j,  0+0j,  1+0j]
        ], order='F', dtype=np.complex128)

        ma02nz('U', 'C', 'N', 1, 3, A)

        np.testing.assert_allclose(A, expected, rtol=1e-14)

    def test_hermitian_upper_permute_adjacent(self):
        """
        Test permutation of adjacent rows/cols 2 and 3 on a 4x4 Hermitian matrix.

        K=2, L=3 (1-based).

        For Hermitian matrix, permuting rows/cols K and L is equivalent to P*A*P^T.
        The upper triangle storage after permutation:
        - Elements before row K are swapped between cols K and L (no conjugation)
        - The (K,L) element is conjugated
        - Elements between K and L are conjugated and swapped
        - Elements after col L are swapped (no conjugation)
        """
        A = np.array([
            [1+0j,   2+1j,   3+2j,   6+4j],
            [0+0j,   4+0j,   5+3j,   8+5j],
            [0+0j,   0+0j,   7+0j,   9+6j],
            [0+0j,   0+0j,   0+0j,  10+0j]
        ], order='F', dtype=np.complex128)

        expected = np.array([
            [1+0j,   3+2j,   2+1j,   6+4j],
            [0+0j,   7+0j,   5-3j,   9+6j],
            [0+0j,   0+0j,   4+0j,   8+5j],
            [0+0j,   0+0j,   0+0j,  10+0j]
        ], order='F', dtype=np.complex128)

        ma02nz('U', 'C', 'N', 2, 3, A)

        np.testing.assert_allclose(A, expected, rtol=1e-14)


class TestMA02NZSkewHermitian:
    """Tests for skew-Hermitian matrix (TRANS='C', SKEW='S')."""

    def test_skew_hermitian_lower_basic(self):
        """
        Test permutation on a 3x3 skew-Hermitian matrix, lower triangle.

        A = -A^H. Diagonal real parts should be zero.
        """
        A = np.array([
            [0+1j,   0+0j,    0+0j],
            [2+1j,   0+3j,    0+0j],
            [4+2j,   5+3j,    0+6j]
        ], order='F', dtype=np.complex128)

        expected = np.array([
            [0+6j,   0+0j,    0+0j],
            [-5+3j,  0+3j,    0+0j],
            [-4+2j,  -2+1j,   0+1j]
        ], order='F', dtype=np.complex128)

        ma02nz('L', 'C', 'S', 1, 3, A)

        np.testing.assert_allclose(A, expected, rtol=1e-14)

    def test_skew_hermitian_upper_basic(self):
        """
        Test permutation on a 3x3 skew-Hermitian matrix, upper triangle.
        """
        A = np.array([
            [0+1j,   2+1j,   4+2j],
            [0+0j,   0+3j,   5+3j],
            [0+0j,   0+0j,   0+6j]
        ], order='F', dtype=np.complex128)

        expected = np.array([
            [0+6j,   -5+3j,   -4+2j],
            [0+0j,   0+3j,    -2+1j],
            [0+0j,   0+0j,    0+1j]
        ], order='F', dtype=np.complex128)

        ma02nz('U', 'C', 'S', 1, 3, A)

        np.testing.assert_allclose(A, expected, rtol=1e-14)


class TestMA02NZSymmetric:
    """Tests for symmetric matrix using transpose (TRANS='T', SKEW='N')."""

    def test_symmetric_lower_basic(self):
        """
        Test permutation on a 3x3 symmetric complex matrix, lower triangle.

        A = A^T (no conjugation).
        """
        A = np.array([
            [1+1j,   0+0j,   0+0j],
            [2+2j,   3+3j,   0+0j],
            [4+4j,   5+5j,   6+6j]
        ], order='F', dtype=np.complex128)

        expected = np.array([
            [6+6j,   0+0j,   0+0j],
            [5+5j,   3+3j,   0+0j],
            [4+4j,   2+2j,   1+1j]
        ], order='F', dtype=np.complex128)

        ma02nz('L', 'T', 'N', 1, 3, A)

        np.testing.assert_allclose(A, expected, rtol=1e-14)

    def test_symmetric_upper_basic(self):
        """
        Test permutation on a 3x3 symmetric complex matrix, upper triangle.
        """
        A = np.array([
            [1+1j,   2+2j,   4+4j],
            [0+0j,   3+3j,   5+5j],
            [0+0j,   0+0j,   6+6j]
        ], order='F', dtype=np.complex128)

        expected = np.array([
            [6+6j,   5+5j,   4+4j],
            [0+0j,   3+3j,   2+2j],
            [0+0j,   0+0j,   1+1j]
        ], order='F', dtype=np.complex128)

        ma02nz('U', 'T', 'N', 1, 3, A)

        np.testing.assert_allclose(A, expected, rtol=1e-14)


class TestMA02NZSkewSymmetric:
    """Tests for skew-symmetric matrix (TRANS='T', SKEW='S')."""

    def test_skew_symmetric_lower_basic(self):
        """
        Test permutation on a 3x3 skew-symmetric matrix, lower triangle.

        A = -A^T (diagonal should be zero).
        """
        A = np.array([
            [0+0j,   0+0j,   0+0j],
            [2+2j,   0+0j,   0+0j],
            [4+4j,   5+5j,   0+0j]
        ], order='F', dtype=np.complex128)

        expected = np.array([
            [0+0j,   0+0j,    0+0j],
            [-5-5j,  0+0j,    0+0j],
            [-4-4j,  -2-2j,   0+0j]
        ], order='F', dtype=np.complex128)

        ma02nz('L', 'T', 'S', 1, 3, A)

        np.testing.assert_allclose(A, expected, rtol=1e-14)

    def test_skew_symmetric_upper_basic(self):
        """
        Test permutation on a 3x3 skew-symmetric matrix, upper triangle.
        """
        A = np.array([
            [0+0j,   2+2j,   4+4j],
            [0+0j,   0+0j,   5+5j],
            [0+0j,   0+0j,   0+0j]
        ], order='F', dtype=np.complex128)

        expected = np.array([
            [0+0j,   -5-5j,   -4-4j],
            [0+0j,   0+0j,    -2-2j],
            [0+0j,   0+0j,    0+0j]
        ], order='F', dtype=np.complex128)

        ma02nz('U', 'T', 'S', 1, 3, A)

        np.testing.assert_allclose(A, expected, rtol=1e-14)


class TestMA02NZMathProperties:
    """Mathematical property validation tests."""

    def test_involution_property(self):
        """
        Mathematical property: Applying same permutation twice returns original.

        Permutation is its own inverse: P * P = I.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n = 5
        real = np.random.randn(n, n)
        imag = np.random.randn(n, n)
        A = (real + 1j * imag).astype(np.complex128, order='F')
        A = np.tril(A)
        for i in range(n):
            A[i, i] = A[i, i].real

        A_orig = A.copy()

        ma02nz('L', 'C', 'N', 2, 4, A)
        ma02nz('L', 'C', 'N', 2, 4, A)

        np.testing.assert_allclose(A, A_orig, rtol=1e-14)

    def test_hermitian_structure_preserved(self):
        """
        Mathematical property: Hermitian structure preserved after permutation.

        Complete the matrix, permute, and verify A = A^H still holds.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n = 4
        real = np.random.randn(n, n)
        imag = np.random.randn(n, n)
        L = (real + 1j * imag).astype(np.complex128, order='F')
        L = np.tril(L)
        for i in range(n):
            L[i, i] = L[i, i].real

        A_full = L + np.tril(L, -1).conj().T

        k, l = 1, 3
        perm = np.eye(n, dtype=np.complex128)
        perm[[k-1, l-1]] = perm[[l-1, k-1]]
        A_permuted = perm @ A_full @ perm.T

        np.testing.assert_allclose(A_permuted, A_permuted.conj().T, rtol=1e-14)


class TestMA02NZEdgeCases:
    """Edge case tests."""

    def test_k_equals_l_no_op(self):
        """
        Test that K=L results in no operation.
        """
        A = np.array([
            [1+0j,  0+0j],
            [2+1j,  3+0j]
        ], order='F', dtype=np.complex128)
        A_orig = A.copy()

        ma02nz('L', 'C', 'N', 2, 2, A)

        np.testing.assert_allclose(A, A_orig, rtol=1e-14)

    def test_k_equals_zero_no_op(self):
        """
        Test that K=0 results in no operation.
        """
        A = np.array([
            [1+0j,  0+0j],
            [2+1j,  3+0j]
        ], order='F', dtype=np.complex128)
        A_orig = A.copy()

        ma02nz('L', 'C', 'N', 0, 2, A)

        np.testing.assert_allclose(A, A_orig, rtol=1e-14)

    def test_n_equals_zero_no_op(self):
        """
        Test that N=0 results in no operation.
        """
        A = np.array([[]], dtype=np.complex128, order='F').reshape(0, 0)
        ma02nz('L', 'C', 'N', 0, 0, A)

    def test_2x2_hermitian_lower(self):
        """
        Test 2x2 Hermitian matrix permutation.
        """
        A = np.array([
            [1+0j,  0+0j],
            [2+1j,  3+0j]
        ], order='F', dtype=np.complex128)

        expected = np.array([
            [3+0j,  0+0j],
            [2-1j,  1+0j]
        ], order='F', dtype=np.complex128)

        ma02nz('L', 'C', 'N', 1, 2, A)

        np.testing.assert_allclose(A, expected, rtol=1e-14)

    def test_5x5_hermitian_random(self):
        """
        Test 5x5 Hermitian matrix permutation with random data.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n = 5
        real = np.random.randn(n, n)
        imag = np.random.randn(n, n)
        L = (real + 1j * imag).astype(np.complex128, order='F')
        L = np.tril(L)
        for i in range(n):
            L[i, i] = L[i, i].real

        A_full = L + np.tril(L, -1).conj().T
        k, l = 2, 5
        perm = np.eye(n, dtype=np.complex128)
        perm[[k-1, l-1]] = perm[[l-1, k-1]]
        expected_full = perm @ A_full @ perm.T
        expected_lower = np.tril(expected_full)

        A = L.copy()
        ma02nz('L', 'C', 'N', k, l, A)

        np.testing.assert_allclose(A, expected_lower, rtol=1e-13)

    def test_5x5_hermitian_upper_random(self):
        """
        Test 5x5 Hermitian matrix permutation with upper storage.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n = 5
        real = np.random.randn(n, n)
        imag = np.random.randn(n, n)
        U = (real + 1j * imag).astype(np.complex128, order='F')
        U = np.triu(U)
        for i in range(n):
            U[i, i] = U[i, i].real

        A_full = U + np.triu(U, 1).conj().T
        k, l = 1, 4
        perm = np.eye(n, dtype=np.complex128)
        perm[[k-1, l-1]] = perm[[l-1, k-1]]
        expected_full = perm @ A_full @ perm.T
        expected_upper = np.triu(expected_full)

        A = U.copy()
        ma02nz('U', 'C', 'N', k, l, A)

        np.testing.assert_allclose(A, expected_upper, rtol=1e-13)
