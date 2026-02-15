"""
Tests for MA02AZ - Complex matrix (conjugate) transposition.

MA02AZ transposes or conjugate-transposes all or part of a two-dimensional
complex matrix A into another matrix B.

TRANS = 'T': transpose B = A^T
TRANS = 'C': conjugate transpose B = A^H (Hermitian transpose)

JOB = 'U': upper triangular part only
JOB = 'L': lower triangular part only
Otherwise: full matrix

Test data sources:
- Mathematical properties of transpose/conjugate-transpose
- Known special cases (Hermitian, symmetric, identity)
"""

import numpy as np
import pytest

from ctrlsys import ma02az


class TestMA02AZBasic:
    """Basic functionality tests for transpose and conjugate transpose."""

    def test_full_transpose(self):
        """
        Test full matrix transpose (TRANS='T', JOB='F').

        B = A^T where A is 2x3 complex matrix.
        """
        a = np.array([
            [1+1j, 2+2j, 3+3j],
            [4+4j, 5+5j, 6+6j]
        ], order='F', dtype=np.complex128)

        b = ma02az('T', 'F', a)

        expected = np.array([
            [1+1j, 4+4j],
            [2+2j, 5+5j],
            [3+3j, 6+6j]
        ], order='F', dtype=np.complex128)

        np.testing.assert_allclose(b, expected, rtol=1e-14)

    def test_full_conjugate_transpose(self):
        """
        Test full matrix conjugate transpose (TRANS='C', JOB='F').

        B = A^H where A is 2x3 complex matrix.
        """
        a = np.array([
            [1+1j, 2+2j, 3+3j],
            [4+4j, 5+5j, 6+6j]
        ], order='F', dtype=np.complex128)

        b = ma02az('C', 'F', a)

        expected = np.array([
            [1-1j, 4-4j],
            [2-2j, 5-5j],
            [3-3j, 6-6j]
        ], order='F', dtype=np.complex128)

        np.testing.assert_allclose(b, expected, rtol=1e-14)

    def test_upper_transpose(self):
        """
        Test upper triangular transpose (TRANS='T', JOB='U').

        Only upper triangular/trapezoidal part is transposed.
        For 3x4 matrix, upper part means i <= j (1-based).
        """
        a = np.array([
            [1+1j, 2+2j, 3+3j, 4+4j],
            [5+5j, 6+6j, 7+7j, 8+8j],
            [9+9j, 10+10j, 11+11j, 12+12j]
        ], order='F', dtype=np.complex128)

        b = ma02az('T', 'U', a)

        # B is 4x3, only positions (j,i) where i <= min(j, m) are set
        # j=1: i=1 -> B[0,0] = A[0,0]
        # j=2: i=1,2 -> B[1,0]=A[0,1], B[1,1]=A[1,1]
        # j=3: i=1,2,3 -> B[2,0]=A[0,2], B[2,1]=A[1,2], B[2,2]=A[2,2]
        # j=4: i=1,2,3 (capped at m=3) -> B[3,0]=A[0,3], B[3,1]=A[1,3], B[3,2]=A[2,3]
        assert b.shape == (4, 3)
        assert b[0, 0] == 1+1j
        assert b[1, 0] == 2+2j
        assert b[1, 1] == 6+6j
        assert b[2, 0] == 3+3j
        assert b[2, 1] == 7+7j
        assert b[2, 2] == 11+11j
        assert b[3, 0] == 4+4j
        assert b[3, 1] == 8+8j
        assert b[3, 2] == 12+12j

    def test_lower_transpose(self):
        """
        Test lower triangular transpose (TRANS='T', JOB='L').

        Only lower triangular/trapezoidal part is transposed.
        For 4x3 matrix, lower part means i >= j.
        """
        a = np.array([
            [1+1j, 2+2j, 3+3j],
            [4+4j, 5+5j, 6+6j],
            [7+7j, 8+8j, 9+9j],
            [10+10j, 11+11j, 12+12j]
        ], order='F', dtype=np.complex128)

        b = ma02az('T', 'L', a)

        # B is 3x4
        # j=1: i=1,2,3,4 -> B[0,0..3]
        # j=2: i=2,3,4 -> B[1,1..3]
        # j=3: i=3,4 -> B[2,2..3]
        assert b.shape == (3, 4)
        assert b[0, 0] == 1+1j
        assert b[0, 1] == 4+4j
        assert b[0, 2] == 7+7j
        assert b[0, 3] == 10+10j
        assert b[1, 1] == 5+5j
        assert b[1, 2] == 8+8j
        assert b[1, 3] == 11+11j
        assert b[2, 2] == 9+9j
        assert b[2, 3] == 12+12j

    def test_upper_conjugate_transpose(self):
        """
        Test upper triangular conjugate transpose (TRANS='C', JOB='U').
        """
        a = np.array([
            [1+1j, 2+2j, 3+3j],
            [4+4j, 5+5j, 6+6j],
            [7+7j, 8+8j, 9+9j]
        ], order='F', dtype=np.complex128)

        b = ma02az('C', 'U', a)

        # Upper part with conjugate
        assert b.shape == (3, 3)
        assert b[0, 0] == 1-1j
        assert b[1, 0] == 2-2j
        assert b[1, 1] == 5-5j
        assert b[2, 0] == 3-3j
        assert b[2, 1] == 6-6j
        assert b[2, 2] == 9-9j

    def test_lower_conjugate_transpose(self):
        """
        Test lower triangular conjugate transpose (TRANS='C', JOB='L').
        """
        a = np.array([
            [1+1j, 2+2j, 3+3j],
            [4+4j, 5+5j, 6+6j],
            [7+7j, 8+8j, 9+9j]
        ], order='F', dtype=np.complex128)

        b = ma02az('C', 'L', a)

        # Lower part with conjugate
        assert b.shape == (3, 3)
        assert b[0, 0] == 1-1j
        assert b[0, 1] == 4-4j
        assert b[0, 2] == 7-7j
        assert b[1, 1] == 5-5j
        assert b[1, 2] == 8-8j
        assert b[2, 2] == 9-9j


class TestMA02AZMathProperties:
    """Mathematical property validation tests."""

    def test_involution_transpose(self):
        """
        Mathematical property: (A^T)^T = A

        Transpose is an involution.
        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        m, n = 4, 5
        real = np.random.randn(m, n)
        imag = np.random.randn(m, n)
        a = (real + 1j * imag).astype(np.complex128, order='F')

        b = ma02az('T', 'F', a)  # B = A^T
        c = ma02az('T', 'F', b)  # C = B^T = (A^T)^T = A

        np.testing.assert_allclose(c, a, rtol=1e-14)

    def test_involution_conjugate_transpose(self):
        """
        Mathematical property: (A^H)^H = A

        Conjugate transpose is an involution.
        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        m, n = 3, 4
        real = np.random.randn(m, n)
        imag = np.random.randn(m, n)
        a = (real + 1j * imag).astype(np.complex128, order='F')

        b = ma02az('C', 'F', a)  # B = A^H
        c = ma02az('C', 'F', b)  # C = B^H = (A^H)^H = A

        np.testing.assert_allclose(c, a, rtol=1e-14)

    def test_hermitian_matrix_conjugate_transpose(self):
        """
        Mathematical property: For Hermitian A, A^H = A.

        A Hermitian matrix is equal to its conjugate transpose.
        """
        # Construct Hermitian matrix: A = A^H
        a = np.array([
            [1+0j, 2+3j, 4-5j],
            [2-3j, 6+0j, 7+8j],
            [4+5j, 7-8j, 9+0j]
        ], order='F', dtype=np.complex128)

        b = ma02az('C', 'F', a)

        np.testing.assert_allclose(b, a, rtol=1e-14)

    def test_real_matrix_transpose_equals_conjugate_transpose(self):
        """
        Mathematical property: For real A, A^T = A^H.

        When imaginary parts are zero, transpose equals conjugate transpose.
        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        m, n = 3, 4
        real = np.random.randn(m, n)
        a = (real + 0j).astype(np.complex128, order='F')

        b_transpose = ma02az('T', 'F', a)
        b_conj_transpose = ma02az('C', 'F', a)

        np.testing.assert_allclose(b_transpose, b_conj_transpose, rtol=1e-14)

    def test_transpose_vs_numpy(self):
        """
        Cross-validation: Compare with NumPy transpose.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        m, n = 5, 3
        real = np.random.randn(m, n)
        imag = np.random.randn(m, n)
        a = (real + 1j * imag).astype(np.complex128, order='F')

        b = ma02az('T', 'F', a)
        expected = np.asfortranarray(a.T)

        np.testing.assert_allclose(b, expected, rtol=1e-14)

    def test_conjugate_transpose_vs_numpy(self):
        """
        Cross-validation: Compare with NumPy conjugate transpose.

        Random seed: 101 (for reproducibility)
        """
        np.random.seed(101)
        m, n = 4, 6
        real = np.random.randn(m, n)
        imag = np.random.randn(m, n)
        a = (real + 1j * imag).astype(np.complex128, order='F')

        b = ma02az('C', 'F', a)
        expected = np.asfortranarray(a.conj().T)

        np.testing.assert_allclose(b, expected, rtol=1e-14)


class TestMA02AZEdgeCases:
    """Edge case tests."""

    def test_square_matrix(self):
        """
        Test with square matrix.
        """
        a = np.array([
            [1+2j, 3+4j],
            [5+6j, 7+8j]
        ], order='F', dtype=np.complex128)

        b = ma02az('C', 'F', a)

        expected = np.array([
            [1-2j, 5-6j],
            [3-4j, 7-8j]
        ], order='F', dtype=np.complex128)

        np.testing.assert_allclose(b, expected, rtol=1e-14)

    def test_single_element(self):
        """
        Test with 1x1 matrix.
        """
        a = np.array([[5+7j]], order='F', dtype=np.complex128)

        b_t = ma02az('T', 'F', a)
        b_c = ma02az('C', 'F', a)

        assert b_t[0, 0] == 5+7j
        assert b_c[0, 0] == 5-7j

    def test_row_vector(self):
        """
        Test with 1xN matrix (row vector).
        """
        a = np.array([[1+1j, 2+2j, 3+3j]], order='F', dtype=np.complex128)

        b = ma02az('C', 'F', a)

        assert b.shape == (3, 1)
        np.testing.assert_allclose(b.flatten(), [1-1j, 2-2j, 3-3j], rtol=1e-14)

    def test_column_vector(self):
        """
        Test with Mx1 matrix (column vector).
        """
        a = np.array([[1+1j], [2+2j], [3+3j]], order='F', dtype=np.complex128)

        b = ma02az('C', 'F', a)

        assert b.shape == (1, 3)
        np.testing.assert_allclose(b.flatten(), [1-1j, 2-2j, 3-3j], rtol=1e-14)

    def test_pure_real(self):
        """
        Test with purely real matrix.
        """
        a = np.array([
            [1+0j, 2+0j],
            [3+0j, 4+0j]
        ], order='F', dtype=np.complex128)

        b_t = ma02az('T', 'F', a)
        b_c = ma02az('C', 'F', a)

        # For real matrix, T and C should give same result
        np.testing.assert_allclose(b_t, b_c, rtol=1e-14)

    def test_pure_imaginary(self):
        """
        Test with purely imaginary matrix.
        """
        a = np.array([
            [0+1j, 0+2j],
            [0+3j, 0+4j]
        ], order='F', dtype=np.complex128)

        b_c = ma02az('C', 'F', a)

        expected = np.array([
            [0-1j, 0-3j],
            [0-2j, 0-4j]
        ], order='F', dtype=np.complex128)

        np.testing.assert_allclose(b_c, expected, rtol=1e-14)

    def test_identity_matrix(self):
        """
        Test with complex identity matrix.
        """
        n = 3
        a = np.eye(n, dtype=np.complex128, order='F')

        b_t = ma02az('T', 'F', a)
        b_c = ma02az('C', 'F', a)

        np.testing.assert_allclose(b_t, a, rtol=1e-14)
        np.testing.assert_allclose(b_c, a, rtol=1e-14)
