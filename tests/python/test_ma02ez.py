"""
Tests for MA02EZ: Store by (skew-)symmetry the upper or lower triangle of a
(skew-)symmetric/Hermitian complex matrix.

MA02EZ handles complex matrices with three mode parameters:
- UPLO: 'U' (upper given) or 'L' (lower given)
- TRANS: 'T' (transpose) or 'C' (conjugate transpose)
- SKEW: 'G' (general), 'N' (symmetric/Hermitian), 'S' (skew-symmetric/Hermitian)

For TRANS='T':
  - SKEW='S': A(i,j) = -A(j,i) (skew-symmetric)
  - Otherwise: A(i,j) = A(j,i) (symmetric)

For TRANS='C':
  - SKEW='G': A(i,j) = conj(A(j,i)), including diagonal
  - SKEW='N': A(i,j) = conj(A(j,i)), diagonal set to real (Hermitian)
  - SKEW='S': A(i,j) = -conj(A(j,i)), diagonal set to pure imaginary (skew-Hermitian)

Random seeds used for reproducibility:
- test_hermitian_lower: 42
- test_hermitian_upper: 123
- test_skew_hermitian_lower: 456
- test_skew_hermitian_upper: 789
- test_symmetric_transpose: 101
- test_skew_symmetric_transpose: 202
- test_hermitian_property: 303
- test_skew_hermitian_property: 404
"""

import numpy as np
import pytest

from ctrlsys import ma02ez


class TestMA02EZHermitian:
    """Tests for Hermitian completion (TRANS='C', SKEW='N')."""

    def test_hermitian_lower_basic(self):
        """
        Test UPLO='L', TRANS='C', SKEW='N' - construct upper from lower for Hermitian.

        Given lower triangle, construct upper as conjugate transpose.
        Diagonal imaginary parts are set to zero.
        """
        A = np.array([
            [1+2j,  0+0j,  0+0j],
            [3+4j,  5+6j,  0+0j],
            [7+8j,  9+10j, 11+12j]
        ], order='F', dtype=np.complex128)

        expected = np.array([
            [1+0j,  3-4j,  7-8j],
            [3+4j,  5+0j,  9-10j],
            [7+8j,  9+10j, 11+0j]
        ], order='F', dtype=np.complex128)

        ma02ez('L', 'C', 'N', A)

        np.testing.assert_allclose(A, expected, rtol=1e-14)

    def test_hermitian_upper_basic(self):
        """
        Test UPLO='U', TRANS='C', SKEW='N' - construct lower from upper for Hermitian.

        Given upper triangle, construct lower as conjugate transpose.
        Diagonal imaginary parts are set to zero.
        """
        A = np.array([
            [1+2j,  3+4j,  7+8j],
            [0+0j,  5+6j,  9+10j],
            [0+0j,  0+0j,  11+12j]
        ], order='F', dtype=np.complex128)

        expected = np.array([
            [1+0j,  3+4j,  7+8j],
            [3-4j,  5+0j,  9+10j],
            [7-8j,  9-10j, 11+0j]
        ], order='F', dtype=np.complex128)

        ma02ez('U', 'C', 'N', A)

        np.testing.assert_allclose(A, expected, rtol=1e-14)

    def test_hermitian_lower_random(self):
        """
        Test Hermitian completion from lower triangle with random data.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n = 4
        real = np.random.randn(n, n)
        imag = np.random.randn(n, n)
        A = (real + 1j * imag).astype(np.complex128, order='F')
        A = np.tril(A)

        ma02ez('L', 'C', 'N', A)

        np.testing.assert_allclose(A, A.conj().T, rtol=1e-14)
        np.testing.assert_allclose(np.diag(A).imag, np.zeros(n), rtol=1e-14)

    def test_hermitian_upper_random(self):
        """
        Test Hermitian completion from upper triangle with random data.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n = 5
        real = np.random.randn(n, n)
        imag = np.random.randn(n, n)
        A = (real + 1j * imag).astype(np.complex128, order='F')
        A = np.triu(A)

        ma02ez('U', 'C', 'N', A)

        np.testing.assert_allclose(A, A.conj().T, rtol=1e-14)
        np.testing.assert_allclose(np.diag(A).imag, np.zeros(n), rtol=1e-14)


class TestMA02EZSkewHermitian:
    """Tests for skew-Hermitian completion (TRANS='C', SKEW='S')."""

    def test_skew_hermitian_lower_basic(self):
        """
        Test UPLO='L', TRANS='C', SKEW='S' - construct upper from lower for skew-Hermitian.

        Given lower triangle, construct upper as negative conjugate transpose.
        Diagonal real parts are set to zero.
        """
        A = np.array([
            [1+2j,  0+0j,  0+0j],
            [3+4j,  5+6j,  0+0j],
            [7+8j,  9+10j, 11+12j]
        ], order='F', dtype=np.complex128)

        expected = np.array([
            [0+2j,  -3+4j,  -7+8j],
            [3+4j,  0+6j,   -9+10j],
            [7+8j,  9+10j,  0+12j]
        ], order='F', dtype=np.complex128)

        ma02ez('L', 'C', 'S', A)

        np.testing.assert_allclose(A, expected, rtol=1e-14)

    def test_skew_hermitian_upper_basic(self):
        """
        Test UPLO='U', TRANS='C', SKEW='S' - construct lower from upper for skew-Hermitian.

        Given upper triangle, construct lower as negative conjugate transpose.
        Diagonal real parts are set to zero.
        """
        A = np.array([
            [1+2j,  3+4j,  7+8j],
            [0+0j,  5+6j,  9+10j],
            [0+0j,  0+0j,  11+12j]
        ], order='F', dtype=np.complex128)

        expected = np.array([
            [0+2j,   3+4j,   7+8j],
            [-3+4j,  0+6j,   9+10j],
            [-7+8j,  -9+10j, 0+12j]
        ], order='F', dtype=np.complex128)

        ma02ez('U', 'C', 'S', A)

        np.testing.assert_allclose(A, expected, rtol=1e-14)

    def test_skew_hermitian_lower_random(self):
        """
        Test skew-Hermitian completion from lower triangle with random data.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n = 4
        real = np.random.randn(n, n)
        imag = np.random.randn(n, n)
        A = (real + 1j * imag).astype(np.complex128, order='F')
        A = np.tril(A)

        ma02ez('L', 'C', 'S', A)

        np.testing.assert_allclose(A, -A.conj().T, rtol=1e-14)
        np.testing.assert_allclose(np.diag(A).real, np.zeros(n), rtol=1e-14)

    def test_skew_hermitian_upper_random(self):
        """
        Test skew-Hermitian completion from upper triangle with random data.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n = 5
        real = np.random.randn(n, n)
        imag = np.random.randn(n, n)
        A = (real + 1j * imag).astype(np.complex128, order='F')
        A = np.triu(A)

        ma02ez('U', 'C', 'S', A)

        np.testing.assert_allclose(A, -A.conj().T, rtol=1e-14)
        np.testing.assert_allclose(np.diag(A).real, np.zeros(n), rtol=1e-14)


class TestMA02EZSymmetric:
    """Tests for symmetric completion using transpose (TRANS='T')."""

    def test_symmetric_lower_basic(self):
        """
        Test UPLO='L', TRANS='T', SKEW='N' - construct upper from lower, symmetric.

        Given lower triangle, construct upper as transpose (no conjugation).
        """
        A = np.array([
            [1+1j,  0+0j,  0+0j],
            [2+2j,  3+3j,  0+0j],
            [4+4j,  5+5j,  6+6j]
        ], order='F', dtype=np.complex128)

        expected = np.array([
            [1+1j,  2+2j,  4+4j],
            [2+2j,  3+3j,  5+5j],
            [4+4j,  5+5j,  6+6j]
        ], order='F', dtype=np.complex128)

        ma02ez('L', 'T', 'N', A)

        np.testing.assert_allclose(A, expected, rtol=1e-14)

    def test_symmetric_upper_basic(self):
        """
        Test UPLO='U', TRANS='T', SKEW='N' - construct lower from upper, symmetric.

        Given upper triangle, construct lower as transpose (no conjugation).
        """
        A = np.array([
            [1+1j,  2+2j,  4+4j],
            [0+0j,  3+3j,  5+5j],
            [0+0j,  0+0j,  6+6j]
        ], order='F', dtype=np.complex128)

        expected = np.array([
            [1+1j,  2+2j,  4+4j],
            [2+2j,  3+3j,  5+5j],
            [4+4j,  5+5j,  6+6j]
        ], order='F', dtype=np.complex128)

        ma02ez('U', 'T', 'N', A)

        np.testing.assert_allclose(A, expected, rtol=1e-14)

    def test_symmetric_transpose_random(self):
        """
        Test symmetric completion using transpose with random data.

        Random seed: 101 (for reproducibility)
        """
        np.random.seed(101)
        n = 4
        real = np.random.randn(n, n)
        imag = np.random.randn(n, n)
        A = (real + 1j * imag).astype(np.complex128, order='F')
        A = np.tril(A)

        ma02ez('L', 'T', 'N', A)

        np.testing.assert_allclose(A, A.T, rtol=1e-14)


class TestMA02EZSkewSymmetric:
    """Tests for skew-symmetric completion using transpose (TRANS='T', SKEW='S')."""

    def test_skew_symmetric_lower_basic(self):
        """
        Test UPLO='L', TRANS='T', SKEW='S' - construct upper from lower, skew-symmetric.

        Given lower triangle, construct upper as negative transpose.
        """
        A = np.array([
            [0+0j,  0+0j,  0+0j],
            [2+2j,  0+0j,  0+0j],
            [4+4j,  5+5j,  0+0j]
        ], order='F', dtype=np.complex128)

        expected = np.array([
            [0+0j,   -2-2j,  -4-4j],
            [2+2j,   0+0j,   -5-5j],
            [4+4j,   5+5j,   0+0j]
        ], order='F', dtype=np.complex128)

        ma02ez('L', 'T', 'S', A)

        np.testing.assert_allclose(A, expected, rtol=1e-14)

    def test_skew_symmetric_upper_basic(self):
        """
        Test UPLO='U', TRANS='T', SKEW='S' - construct lower from upper, skew-symmetric.

        Given upper triangle, construct lower as negative transpose.
        """
        A = np.array([
            [0+0j,   2+2j,   4+4j],
            [0+0j,   0+0j,   5+5j],
            [0+0j,   0+0j,   0+0j]
        ], order='F', dtype=np.complex128)

        expected = np.array([
            [0+0j,   2+2j,   4+4j],
            [-2-2j,  0+0j,   5+5j],
            [-4-4j,  -5-5j,  0+0j]
        ], order='F', dtype=np.complex128)

        ma02ez('U', 'T', 'S', A)

        np.testing.assert_allclose(A, expected, rtol=1e-14)

    def test_skew_symmetric_transpose_random(self):
        """
        Test skew-symmetric completion using transpose with random data.

        Random seed: 202 (for reproducibility)
        """
        np.random.seed(202)
        n = 4
        real = np.random.randn(n, n)
        imag = np.random.randn(n, n)
        A = (real + 1j * imag).astype(np.complex128, order='F')
        A = np.tril(A, k=-1)  # Strict lower

        ma02ez('L', 'T', 'S', A)

        np.testing.assert_allclose(A, -A.T, rtol=1e-14)


class TestMA02EZGeneral:
    """Tests for general case (TRANS='C', SKEW='G')."""

    def test_general_lower_basic(self):
        """
        Test UPLO='L', TRANS='C', SKEW='G' - general conjugate transpose.

        Construct upper from lower including diagonal (full conjugate transpose).
        """
        A = np.array([
            [1+2j,  0+0j,  0+0j],
            [3+4j,  5+6j,  0+0j],
            [7+8j,  9+10j, 11+12j]
        ], order='F', dtype=np.complex128)

        expected = np.array([
            [1-2j,  3-4j,  7-8j],
            [3+4j,  5-6j,  9-10j],
            [7+8j,  9+10j, 11-12j]
        ], order='F', dtype=np.complex128)

        ma02ez('L', 'C', 'G', A)

        np.testing.assert_allclose(A, expected, rtol=1e-14)

    def test_general_upper_basic(self):
        """
        Test UPLO='U', TRANS='C', SKEW='G' - general conjugate transpose.

        Construct lower from upper including diagonal.
        """
        A = np.array([
            [1+2j,  3+4j,  7+8j],
            [0+0j,  5+6j,  9+10j],
            [0+0j,  0+0j,  11+12j]
        ], order='F', dtype=np.complex128)

        expected = np.array([
            [1-2j,  3+4j,  7+8j],
            [3-4j,  5-6j,  9+10j],
            [7-8j,  9-10j, 11-12j]
        ], order='F', dtype=np.complex128)

        ma02ez('U', 'C', 'G', A)

        np.testing.assert_allclose(A, expected, rtol=1e-14)


class TestMA02EZMathProperties:
    """Mathematical property validation tests."""

    def test_hermitian_property(self):
        """
        Mathematical property: A = A^H after Hermitian completion.

        Random seed: 303 (for reproducibility)
        """
        np.random.seed(303)
        n = 5
        real = np.random.randn(n, n)
        imag = np.random.randn(n, n)
        A = (real + 1j * imag).astype(np.complex128, order='F')
        A = np.tril(A)

        ma02ez('L', 'C', 'N', A)

        np.testing.assert_allclose(A, A.conj().T, rtol=1e-14)

    def test_skew_hermitian_property(self):
        """
        Mathematical property: A = -A^H after skew-Hermitian completion.

        Random seed: 404 (for reproducibility)
        """
        np.random.seed(404)
        n = 5
        real = np.random.randn(n, n)
        imag = np.random.randn(n, n)
        A = (real + 1j * imag).astype(np.complex128, order='F')
        A = np.triu(A)

        ma02ez('U', 'C', 'S', A)

        np.testing.assert_allclose(A, -A.conj().T, rtol=1e-14)

    def test_symmetric_property(self):
        """
        Mathematical property: A = A^T after symmetric completion (TRANS='T').

        Random seed: 505 (for reproducibility)
        """
        np.random.seed(505)
        n = 4
        real = np.random.randn(n, n)
        imag = np.random.randn(n, n)
        A = (real + 1j * imag).astype(np.complex128, order='F')
        A = np.triu(A)

        ma02ez('U', 'T', 'N', A)

        np.testing.assert_allclose(A, A.T, rtol=1e-14)

    def test_skew_symmetric_property(self):
        """
        Mathematical property: A = -A^T after skew-symmetric completion (TRANS='T', SKEW='S').

        Random seed: 606 (for reproducibility)
        """
        np.random.seed(606)
        n = 4
        real = np.random.randn(n, n)
        imag = np.random.randn(n, n)
        A = (real + 1j * imag).astype(np.complex128, order='F')
        A = np.triu(A, k=1)  # Strict upper

        ma02ez('U', 'T', 'S', A)

        np.testing.assert_allclose(A, -A.T, rtol=1e-14)


class TestMA02EZEdgeCases:
    """Edge case tests."""

    def test_n_one(self):
        """
        Test with N=1 (single element).

        For Hermitian: diagonal should become real.
        For skew-Hermitian: diagonal should become pure imaginary.
        """
        A1 = np.array([[3+4j]], dtype=np.complex128, order='F')
        ma02ez('U', 'C', 'N', A1)
        assert A1[0, 0] == 3+0j

        A2 = np.array([[3+4j]], dtype=np.complex128, order='F')
        ma02ez('L', 'C', 'S', A2)
        assert A2[0, 0] == 0+4j

        A3 = np.array([[3+4j]], dtype=np.complex128, order='F')
        ma02ez('U', 'C', 'G', A3)
        assert A3[0, 0] == 3-4j

    def test_n_two(self):
        """
        Test with N=2.
        """
        A = np.array([
            [1+1j, 2+3j],
            [4+5j, 6+7j]
        ], dtype=np.complex128, order='F')

        expected = np.array([
            [1+0j, 2+3j],
            [2-3j, 6+0j]
        ], dtype=np.complex128, order='F')

        ma02ez('U', 'C', 'N', A)

        np.testing.assert_allclose(A, expected, rtol=1e-14)

    def test_invalid_uplo_no_op(self):
        """
        Test that invalid UPLO does not modify the matrix.
        """
        np.random.seed(707)
        n = 3
        real = np.random.randn(n, n)
        imag = np.random.randn(n, n)
        A = (real + 1j * imag).astype(np.complex128, order='F')
        A_orig = A.copy()

        ma02ez('X', 'C', 'N', A)

        np.testing.assert_allclose(A, A_orig, rtol=1e-14)

    def test_pure_real_matrix(self):
        """
        Test with purely real input matrix.
        """
        A = np.array([
            [1+0j, 0+0j, 0+0j],
            [2+0j, 3+0j, 0+0j],
            [4+0j, 5+0j, 6+0j]
        ], dtype=np.complex128, order='F')

        ma02ez('L', 'C', 'N', A)

        np.testing.assert_allclose(A, A.conj().T, rtol=1e-14)
        np.testing.assert_allclose(A.imag, np.zeros((3, 3)), rtol=1e-14)

    def test_pure_imaginary_matrix(self):
        """
        Test with purely imaginary input matrix for skew-Hermitian.
        """
        A = np.array([
            [0+1j, 0+0j, 0+0j],
            [0+2j, 0+3j, 0+0j],
            [0+4j, 0+5j, 0+6j]
        ], dtype=np.complex128, order='F')

        ma02ez('L', 'C', 'S', A)

        np.testing.assert_allclose(A, -A.conj().T, rtol=1e-14)
        np.testing.assert_allclose(np.diag(A).real, np.zeros(3), rtol=1e-14)
