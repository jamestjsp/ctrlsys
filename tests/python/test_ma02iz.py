"""
Tests for MA02IZ - Complex skew-Hamiltonian/Hamiltonian matrix norm computation.

MA02IZ computes the one norm, Frobenius norm, infinity norm, or max element
of a complex skew-Hamiltonian or Hamiltonian matrix:

    Skew-Hamiltonian:  X = [A, G; Q, A^H]   where G = -G^H, Q = -Q^H
    Hamiltonian:       X = [A, G; Q, -A^H]  where G = G^H, Q = Q^H

For these matrix types, the infinity norm equals the one norm.
"""

import numpy as np
import pytest
from ctrlsys import ma02iz


def build_skew_hamiltonian(a, qg, n):
    """Build full 2n x 2n skew-Hamiltonian matrix from packed representation.

    Skew-Hamiltonian: X = [A, G; Q, A^H] where G = -G^H, Q = -Q^H
    """
    x = np.zeros((2*n, 2*n), dtype=complex, order='F')

    x[:n, :n] = a
    x[n:, n:] = a.conj().T

    q = np.zeros((n, n), dtype=complex, order='F')
    for j in range(n):
        for i in range(j+1, n):
            q[i, j] = qg[i, j]
            q[j, i] = -qg[i, j].conj()
        q[j, j] = 1j * qg[j, j].imag

    g = np.zeros((n, n), dtype=complex, order='F')
    for j in range(n):
        for i in range(j):
            g[i, j] = qg[i, j+1]
            g[j, i] = -qg[i, j+1].conj()
        g[j, j] = 1j * qg[j, j+1].imag

    x[n:, :n] = q
    x[:n, n:] = g

    return x


def build_hamiltonian(a, qg, n):
    """Build full 2n x 2n Hamiltonian matrix from packed representation.

    Hamiltonian: X = [A, G; Q, -A^H] where G = G^H, Q = Q^H
    """
    x = np.zeros((2*n, 2*n), dtype=complex, order='F')

    x[:n, :n] = a
    x[n:, n:] = -a.conj().T

    q = np.zeros((n, n), dtype=complex, order='F')
    for j in range(n):
        for i in range(j+1, n):
            q[i, j] = qg[i, j]
            q[j, i] = qg[i, j].conj()
        q[j, j] = qg[j, j].real

    g = np.zeros((n, n), dtype=complex, order='F')
    for j in range(n):
        for i in range(j):
            g[i, j] = qg[i, j+1]
            g[j, i] = qg[i, j+1].conj()
        g[j, j] = qg[j, j+1].real

    x[n:, :n] = q
    x[:n, n:] = g

    return x


class TestMA02IZBasic:
    """Basic functionality tests."""

    def test_empty_matrix(self):
        """Test n=0 returns zero."""
        a = np.zeros((0, 0), dtype=complex, order='F')
        qg = np.zeros((0, 1), dtype=complex, order='F')

        for typ in ['S', 'H']:
            for norm in ['M', '1', 'O', 'I', 'F', 'E']:
                result = ma02iz(typ, norm, a, qg)
                assert result == 0.0

    def test_hamiltonian_max_norm_1x1(self):
        """Test max norm for 1x1 Hamiltonian matrix.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n = 1
        a = np.array([[2.0 + 3.0j]], dtype=complex, order='F')
        qg = np.array([[1.5, 2.5]], dtype=complex, order='F')

        x = build_hamiltonian(a, qg, n)
        expected = np.max(np.abs(x))

        result = ma02iz('H', 'M', a, qg)
        np.testing.assert_allclose(result, expected, rtol=1e-14)

    def test_skew_hamiltonian_max_norm_2x2(self):
        """Test max norm for 2x2 skew-Hamiltonian matrix.

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n = 2
        a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
        qg = (np.random.randn(n, n+1) + 1j * np.random.randn(n, n+1)).astype(complex, order='F')

        x = build_skew_hamiltonian(a, qg, n)
        expected = np.max(np.abs(x))

        result = ma02iz('S', 'M', a, qg)
        np.testing.assert_allclose(result, expected, rtol=1e-14)


class TestMA02IZOneNorm:
    """Tests for one norm computation."""

    def test_hamiltonian_one_norm_2x2(self):
        """Test one norm for 2x2 Hamiltonian matrix.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n = 2
        a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
        qg = (np.random.randn(n, n+1) + 1j * np.random.randn(n, n+1)).astype(complex, order='F')

        x = build_hamiltonian(a, qg, n)
        expected = np.linalg.norm(x, ord=1)

        result = ma02iz('H', '1', a, qg)
        np.testing.assert_allclose(result, expected, rtol=1e-14)

    def test_skew_hamiltonian_one_norm_3x3(self):
        """Test one norm for 3x3 skew-Hamiltonian matrix.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n = 3
        a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
        qg = (np.random.randn(n, n+1) + 1j * np.random.randn(n, n+1)).astype(complex, order='F')

        x = build_skew_hamiltonian(a, qg, n)
        expected = np.linalg.norm(x, ord=1)

        result = ma02iz('S', '1', a, qg)
        np.testing.assert_allclose(result, expected, rtol=1e-14)

    def test_one_equals_infinity_norm(self):
        """Mathematical property: for (skew-)Hamiltonian, 1-norm equals inf-norm.

        Random seed: 111 (for reproducibility)
        """
        np.random.seed(111)
        n = 4
        a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
        qg = (np.random.randn(n, n+1) + 1j * np.random.randn(n, n+1)).astype(complex, order='F')

        for typ in ['S', 'H']:
            one_norm = ma02iz(typ, '1', a, qg)
            inf_norm = ma02iz(typ, 'I', a, qg)
            np.testing.assert_allclose(one_norm, inf_norm, rtol=1e-14)


class TestMA02IZFrobeniusNorm:
    """Tests for Frobenius norm computation."""

    def test_hamiltonian_frobenius_norm_2x2(self):
        """Test Frobenius norm for 2x2 Hamiltonian matrix.

        Random seed: 222 (for reproducibility)
        """
        np.random.seed(222)
        n = 2
        a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
        qg = (np.random.randn(n, n+1) + 1j * np.random.randn(n, n+1)).astype(complex, order='F')

        x = build_hamiltonian(a, qg, n)
        expected = np.linalg.norm(x, 'fro')

        result = ma02iz('H', 'F', a, qg)
        np.testing.assert_allclose(result, expected, rtol=1e-13)

    def test_skew_hamiltonian_frobenius_norm_3x3(self):
        """Test Frobenius norm for 3x3 skew-Hamiltonian matrix.

        Random seed: 333 (for reproducibility)
        """
        np.random.seed(333)
        n = 3
        a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
        qg = (np.random.randn(n, n+1) + 1j * np.random.randn(n, n+1)).astype(complex, order='F')

        x = build_skew_hamiltonian(a, qg, n)
        expected = np.linalg.norm(x, 'fro')

        result = ma02iz('S', 'F', a, qg)
        np.testing.assert_allclose(result, expected, rtol=1e-13)

    def test_frobenius_norm_e_variant(self):
        """Test that 'E' gives same result as 'F' for Frobenius norm.

        Random seed: 444 (for reproducibility)
        """
        np.random.seed(444)
        n = 3
        a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
        qg = (np.random.randn(n, n+1) + 1j * np.random.randn(n, n+1)).astype(complex, order='F')

        for typ in ['S', 'H']:
            f_norm = ma02iz(typ, 'F', a, qg)
            e_norm = ma02iz(typ, 'E', a, qg)
            np.testing.assert_allclose(f_norm, e_norm, rtol=1e-14)


class TestMA02IZNormEquivalence:
    """Tests for norm equivalence properties."""

    def test_o_equals_1_norm(self):
        """Test that 'O' gives same result as '1' for one norm.

        Random seed: 555 (for reproducibility)
        """
        np.random.seed(555)
        n = 3
        a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
        qg = (np.random.randn(n, n+1) + 1j * np.random.randn(n, n+1)).astype(complex, order='F')

        for typ in ['S', 'H']:
            one_norm = ma02iz(typ, '1', a, qg)
            o_norm = ma02iz(typ, 'O', a, qg)
            np.testing.assert_allclose(one_norm, o_norm, rtol=1e-14)


class TestMA02IZLargerMatrices:
    """Tests with larger matrices to verify algorithm correctness."""

    def test_hamiltonian_all_norms_5x5(self):
        """Test all norms for 5x5 Hamiltonian matrix.

        Random seed: 666 (for reproducibility)
        """
        np.random.seed(666)
        n = 5
        a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
        qg = (np.random.randn(n, n+1) + 1j * np.random.randn(n, n+1)).astype(complex, order='F')

        x = build_hamiltonian(a, qg, n)

        max_expected = np.max(np.abs(x))
        one_expected = np.linalg.norm(x, ord=1)
        fro_expected = np.linalg.norm(x, 'fro')

        max_result = ma02iz('H', 'M', a, qg)
        one_result = ma02iz('H', '1', a, qg)
        fro_result = ma02iz('H', 'F', a, qg)

        np.testing.assert_allclose(max_result, max_expected, rtol=1e-14)
        np.testing.assert_allclose(one_result, one_expected, rtol=1e-14)
        np.testing.assert_allclose(fro_result, fro_expected, rtol=1e-13)

    def test_skew_hamiltonian_all_norms_5x5(self):
        """Test all norms for 5x5 skew-Hamiltonian matrix.

        Random seed: 777 (for reproducibility)
        """
        np.random.seed(777)
        n = 5
        a = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(complex, order='F')
        qg = (np.random.randn(n, n+1) + 1j * np.random.randn(n, n+1)).astype(complex, order='F')

        x = build_skew_hamiltonian(a, qg, n)

        max_expected = np.max(np.abs(x))
        one_expected = np.linalg.norm(x, ord=1)
        fro_expected = np.linalg.norm(x, 'fro')

        max_result = ma02iz('S', 'M', a, qg)
        one_result = ma02iz('S', '1', a, qg)
        fro_result = ma02iz('S', 'F', a, qg)

        np.testing.assert_allclose(max_result, max_expected, rtol=1e-14)
        np.testing.assert_allclose(one_result, one_expected, rtol=1e-14)
        np.testing.assert_allclose(fro_result, fro_expected, rtol=1e-13)
