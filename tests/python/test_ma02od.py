"""
Tests for MA02OD: Count zero rows of a real (skew-)Hamiltonian matrix.

The matrix H has the form:
    (  A    D   )
H = (           )
    (  E  +/-A' )

where A is M-by-M, D is symmetric/skew-symmetric (upper triangle in DE columns 2..M+1),
and E is symmetric/skew-symmetric (lower triangle in DE column 1..M).

For Hamiltonian (SKEW='H'): H = [A, D; E, A'] with D=D', E=E' (diagonal included)
For skew-Hamiltonian (SKEW='S'): H = [A, D; E, -A'] with D=-D', E=-E' (diagonal zero)
"""

import numpy as np
import pytest
from ctrlsys import ma02od


class TestMA02ODBasic:
    """Basic functionality tests."""

    def test_all_zero_hamiltonian(self):
        """
        Test with all-zero Hamiltonian matrix.

        Random seed: N/A (deterministic zeros)
        """
        m = 3
        a = np.zeros((m, m), order='F', dtype=float)
        de = np.zeros((m, m + 1), order='F', dtype=float)

        nz = ma02od('H', a, de)

        assert nz == 2 * m

    def test_all_zero_skew_hamiltonian(self):
        """
        Test with all-zero skew-Hamiltonian matrix.

        Random seed: N/A (deterministic zeros)
        """
        m = 3
        a = np.zeros((m, m), order='F', dtype=float)
        de = np.zeros((m, m + 1), order='F', dtype=float)

        nz = ma02od('S', a, de)

        assert nz == 2 * m

    def test_no_zero_rows_hamiltonian(self):
        """
        Test with non-zero Hamiltonian matrix (no zero rows expected).

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        m = 3
        a = np.random.randn(m, m).astype(float, order='F')
        de = np.random.randn(m, m + 1).astype(float, order='F')

        nz = ma02od('H', a, de)

        assert nz == 0

    def test_no_zero_rows_skew_hamiltonian(self):
        """
        Test with non-zero skew-Hamiltonian matrix.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        m = 3
        a = np.random.randn(m, m).astype(float, order='F')
        de = np.random.randn(m, m + 1).astype(float, order='F')

        nz = ma02od('S', a, de)

        assert nz == 0


class TestMA02ODPartialZeros:
    """Test cases with partial zero rows."""

    def test_m_equals_1_hamiltonian(self):
        """
        Test with m=1 (smallest non-trivial case).

        2x2 Hamiltonian matrix:
        H = [a11, d11; e11, a11]

        Random seed: N/A (deterministic)
        """
        m = 1
        a = np.array([[0.0]], order='F', dtype=float)
        de = np.array([[0.0, 0.0]], order='F', dtype=float)

        nz = ma02od('H', a, de)

        assert nz == 2

    def test_m_equals_1_nonzero(self):
        """
        Test with m=1, non-zero entries.

        Random seed: N/A (deterministic)
        """
        m = 1
        a = np.array([[1.0]], order='F', dtype=float)
        de = np.array([[0.0, 0.0]], order='F', dtype=float)

        nz = ma02od('H', a, de)

        assert nz == 0


class TestMA02ODSkewHamiltonian:
    """Test skew-Hamiltonian specific behavior."""

    def test_skew_diagonal_ignored(self):
        """
        Test that for skew-Hamiltonian, diagonal of DE is ignored.

        For SKEW='S', the diagonal and first superdiagonal of DE
        are assumed zero (not referenced).

        Random seed: N/A (deterministic)
        """
        m = 3
        a = np.zeros((m, m), order='F', dtype=float)
        de = np.zeros((m, m + 1), order='F', dtype=float)

        for i in range(m):
            de[i, i] = 999.0
            if i < m:
                de[i, i + 1] = 999.0

        nz = ma02od('S', a, de)

        assert nz == 2 * m


class TestMA02ODEdgeCases:
    """Edge case tests."""

    def test_m_equals_0(self):
        """
        Test with m=0 (empty matrix).

        Random seed: N/A (empty case)
        """
        m = 0
        a = np.zeros((0, 0), order='F', dtype=float)
        de = np.zeros((0, 1), order='F', dtype=float)

        nz = ma02od('H', a, de)

        assert nz == 0

    def test_lowercase_skew(self):
        """
        Test that lowercase 'h' and 's' work.

        Random seed: N/A (deterministic zeros)
        """
        m = 2
        a = np.zeros((m, m), order='F', dtype=float)
        de = np.zeros((m, m + 1), order='F', dtype=float)

        nz_h = ma02od('h', a, de)
        nz_H = ma02od('H', a, de)
        nz_s = ma02od('s', a, de)
        nz_S = ma02od('S', a, de)

        assert nz_h == nz_H == 2 * m
        assert nz_s == nz_S == 2 * m


class TestMA02ODMathematical:
    """Mathematical property validation tests."""

    def test_count_bounded(self):
        """
        Validate that zero row count is bounded by 2*M.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        m = 5
        a = np.random.randn(m, m).astype(float, order='F')
        de = np.random.randn(m, m + 1).astype(float, order='F')

        nz = ma02od('H', a, de)

        assert 0 <= nz <= 2 * m

        a_zero = np.zeros((m, m), order='F', dtype=float)
        de_zero = np.zeros((m, m + 1), order='F', dtype=float)
        nz_max = ma02od('H', a_zero, de_zero)

        assert nz_max == 2 * m
