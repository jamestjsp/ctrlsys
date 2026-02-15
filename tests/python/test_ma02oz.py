"""
Tests for MA02OZ: Count zero rows of a complex (skew-)Hamiltonian matrix.

The matrix H has the form:
    (  A    D   )
H = (           )
    (  E  +/-A' )

where A is M-by-M complex, D is (skew-)Hermitian (upper triangle in DE columns 2..M+1),
and E is (skew-)Hermitian (lower triangle in DE columns 1..M).

For Hamiltonian (SKEW='H'): H = [A, D; E, A'] with D=D^H, E=E^H
    - Real parts of diagonal entries of DE are used
For skew-Hamiltonian (SKEW='S'): H = [A, D; E, -A'] with D=-D^H, E=-E^H
    - Imaginary parts of diagonal entries of DE are used (real parts assumed zero)

This is the complex version of MA02OD.
"""

import numpy as np
import pytest
from ctrlsys import ma02oz


class TestMA02OZBasic:
    """Basic functionality tests."""

    def test_all_zero_hamiltonian(self):
        """
        Test with all-zero Hamiltonian matrix.

        Random seed: N/A (deterministic zeros)
        """
        m = 3
        a = np.zeros((m, m), order='F', dtype=complex)
        de = np.zeros((m, m + 1), order='F', dtype=complex)

        nz = ma02oz('H', a, de)

        assert nz == 2 * m

    def test_all_zero_skew_hamiltonian(self):
        """
        Test with all-zero skew-Hamiltonian matrix.

        Random seed: N/A (deterministic zeros)
        """
        m = 3
        a = np.zeros((m, m), order='F', dtype=complex)
        de = np.zeros((m, m + 1), order='F', dtype=complex)

        nz = ma02oz('S', a, de)

        assert nz == 2 * m

    def test_no_zero_rows_hamiltonian(self):
        """
        Test with non-zero Hamiltonian matrix (no zero rows expected).

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        m = 3
        a = (np.random.randn(m, m) + 1j * np.random.randn(m, m)).astype(complex, order='F')
        de = (np.random.randn(m, m + 1) + 1j * np.random.randn(m, m + 1)).astype(complex, order='F')

        nz = ma02oz('H', a, de)

        assert nz == 0

    def test_no_zero_rows_skew_hamiltonian(self):
        """
        Test with non-zero skew-Hamiltonian matrix.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        m = 3
        a = (np.random.randn(m, m) + 1j * np.random.randn(m, m)).astype(complex, order='F')
        de = (np.random.randn(m, m + 1) + 1j * np.random.randn(m, m + 1)).astype(complex, order='F')

        nz = ma02oz('S', a, de)

        assert nz == 0


class TestMA02OZPartialZeros:
    """Test cases with partial zero rows."""

    def test_m_equals_1_hamiltonian(self):
        """
        Test with m=1 (smallest non-trivial case).

        2x2 Hamiltonian matrix:
        H = [a11, d11; e11, a11^H]

        Random seed: N/A (deterministic)
        """
        m = 1
        a = np.array([[0.0 + 0.0j]], order='F', dtype=complex)
        de = np.array([[0.0 + 0.0j, 0.0 + 0.0j]], order='F', dtype=complex)

        nz = ma02oz('H', a, de)

        assert nz == 2

    def test_m_equals_1_nonzero(self):
        """
        Test with m=1, non-zero entries.

        Random seed: N/A (deterministic)
        """
        m = 1
        a = np.array([[1.0 + 2.0j]], order='F', dtype=complex)
        de = np.array([[0.0 + 0.0j, 0.0 + 0.0j]], order='F', dtype=complex)

        nz = ma02oz('H', a, de)

        assert nz == 0

    def test_one_zero_row_in_a(self):
        """
        Test matrix with one row of A being zero.

        Random seed: N/A (deterministic)
        """
        m = 2
        a = np.array([
            [0.0 + 0.0j, 0.0 + 0.0j],
            [1.0 + 1.0j, 2.0 - 1.0j]
        ], order='F', dtype=complex)
        de = np.zeros((m, m + 1), order='F', dtype=complex)

        nz = ma02oz('H', a, de)

        assert nz >= 1


class TestMA02OZSkewHamiltonian:
    """Test skew-Hamiltonian specific behavior."""

    def test_skew_diagonal_imaginary_part_used(self):
        """
        Test that for skew-Hamiltonian, imaginary part of diagonal of DE is used.

        For SKEW='S', the real parts of diagonal and first superdiagonal
        are assumed zero (not referenced). Only imaginary parts matter.

        Random seed: N/A (deterministic)
        """
        m = 3
        a = np.zeros((m, m), order='F', dtype=complex)
        de = np.zeros((m, m + 1), order='F', dtype=complex)

        for i in range(m):
            de[i, i] = 999.0 + 0.0j
            if i < m:
                de[i, i + 1] = 999.0 + 0.0j

        nz = ma02oz('S', a, de)

        assert nz == 2 * m

    def test_skew_imaginary_diagonal_nonzero(self):
        """
        Test that imaginary diagonal makes row non-zero for skew-Hamiltonian.

        Random seed: N/A (deterministic)
        """
        m = 1
        a = np.zeros((m, m), order='F', dtype=complex)
        de = np.array([[0.0 + 1.0j, 0.0 + 0.0j]], order='F', dtype=complex)

        nz = ma02oz('S', a, de)

        assert nz == 1


class TestMA02OZHamiltonian:
    """Test Hamiltonian specific behavior."""

    def test_hamiltonian_real_diagonal_used(self):
        """
        Test that for Hamiltonian, real part of diagonal of DE is used.

        For SKEW='H', the imaginary parts of diagonal and first superdiagonal
        are assumed zero (not referenced). Only real parts matter.

        Random seed: N/A (deterministic)
        """
        m = 3
        a = np.zeros((m, m), order='F', dtype=complex)
        de = np.zeros((m, m + 1), order='F', dtype=complex)

        for i in range(m):
            de[i, i] = 0.0 + 999.0j
            if i < m:
                de[i, i + 1] = 0.0 + 999.0j

        nz = ma02oz('H', a, de)

        assert nz == 2 * m

    def test_hamiltonian_real_diagonal_nonzero(self):
        """
        Test that real diagonal makes row non-zero for Hamiltonian.

        Random seed: N/A (deterministic)
        """
        m = 1
        a = np.zeros((m, m), order='F', dtype=complex)
        de = np.array([[1.0 + 0.0j, 0.0 + 0.0j]], order='F', dtype=complex)

        nz = ma02oz('H', a, de)

        assert nz == 1


class TestMA02OZEdgeCases:
    """Edge case tests."""

    def test_m_equals_0(self):
        """
        Test with m=0 (empty matrix).

        Random seed: N/A (empty case)
        """
        m = 0
        a = np.zeros((0, 0), order='F', dtype=complex)
        de = np.zeros((0, 1), order='F', dtype=complex)

        nz = ma02oz('H', a, de)

        assert nz == 0

    def test_lowercase_skew(self):
        """
        Test that lowercase 'h' and 's' work.

        Random seed: N/A (deterministic zeros)
        """
        m = 2
        a = np.zeros((m, m), order='F', dtype=complex)
        de = np.zeros((m, m + 1), order='F', dtype=complex)

        nz_h = ma02oz('h', a, de)
        nz_H = ma02oz('H', a, de)
        nz_s = ma02oz('s', a, de)
        nz_S = ma02oz('S', a, de)

        assert nz_h == nz_H == 2 * m
        assert nz_s == nz_S == 2 * m


class TestMA02OZMathematical:
    """Mathematical property validation tests."""

    def test_count_bounded(self):
        """
        Validate that zero row count is bounded by 2*M.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        m = 5
        a = (np.random.randn(m, m) + 1j * np.random.randn(m, m)).astype(complex, order='F')
        de = (np.random.randn(m, m + 1) + 1j * np.random.randn(m, m + 1)).astype(complex, order='F')

        nz = ma02oz('H', a, de)

        assert 0 <= nz <= 2 * m

        a_zero = np.zeros((m, m), order='F', dtype=complex)
        de_zero = np.zeros((m, m + 1), order='F', dtype=complex)
        nz_max = ma02oz('H', a_zero, de_zero)

        assert nz_max == 2 * m

    def test_complex_vs_real_consistency(self):
        """
        Test that for real-valued inputs, MA02OZ behaves like MA02OD.

        Random seed: 456 (for reproducibility)
        """
        from ctrlsys import ma02od

        np.random.seed(456)
        m = 3

        a_real = np.random.randn(m, m).astype(float, order='F')
        de_real = np.random.randn(m, m + 1).astype(float, order='F')

        a_complex = a_real.astype(complex, order='F')
        de_complex = de_real.astype(complex, order='F')

        nz_real = ma02od('H', a_real, de_real)
        nz_complex = ma02oz('H', a_complex, de_complex)

        assert nz_real == nz_complex
