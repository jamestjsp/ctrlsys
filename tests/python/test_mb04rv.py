"""
Tests for mb04rv - Complex generalized Sylvester equation solver

Solves:
    A * R - L * B = scale * C
    D * R - L * E = scale * F

where A, B, D, E are upper triangular complex matrices (generalized Schur form).
Solution (R, L) overwrites (C, F).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


def cabs1(z):
    """Taxi-cab norm: |real| + |imag| (as used in SLICOT)"""
    return np.abs(z.real) + np.abs(z.imag)


class TestMB04RV:
    """Tests for mb04rv complex generalized Sylvester solver."""

    def test_basic_1x1(self):
        """
        Test 1x1 system (simplest case).

        Random seed: 42 (for reproducibility)

        Eigenvalues of (A,D) and (B,E) must be distinct.
        (A,D) has eigenvalue A/D = (2+1j)/1 = 2+1j
        (B,E) has eigenvalue B/E = (3+0.5j)/1 = 3+0.5j
        These are distinct, so system is not singular.
        """
        from ctrlsys import mb04rv

        np.random.seed(42)

        m, n = 1, 1
        pmax = 1e6

        a = np.array([[2.0 + 1.0j]], dtype=complex, order='F')
        b = np.array([[3.0 + 0.5j]], dtype=complex, order='F')
        d = np.array([[1.0 + 0.0j]], dtype=complex, order='F')
        e = np.array([[1.0 + 0.0j]], dtype=complex, order='F')

        c = np.array([[3.0 + 2.0j]], dtype=complex, order='F')
        f = np.array([[1.0 + 1.0j]], dtype=complex, order='F')
        c_orig = c.copy()
        f_orig = f.copy()

        r, l, scale, info = mb04rv(m, n, pmax, a, b, c, d, e, f)

        assert info == 0
        assert 0.0 < scale <= 1.0

        residual1 = a @ r - l @ b - scale * c_orig
        residual2 = d @ r - l @ e - scale * f_orig
        assert_allclose(residual1, 0.0, atol=1e-14)
        assert_allclose(residual2, 0.0, atol=1e-14)

    def test_2x2_system(self):
        """
        Test 2x2 upper triangular system.

        Random seed: 123 (for reproducibility)
        """
        from ctrlsys import mb04rv

        np.random.seed(123)

        m, n = 2, 2
        pmax = 1e6

        a = np.array([[2.0+1.0j, 0.5+0.2j],
                      [0.0+0.0j, 1.0+0.5j]], dtype=complex, order='F')
        b = np.array([[1.0+0.3j, 0.3+0.1j],
                      [0.0+0.0j, 0.8+0.2j]], dtype=complex, order='F')
        d = np.array([[1.0+0.0j, 0.2+0.1j],
                      [0.0+0.0j, 0.5+0.0j]], dtype=complex, order='F')
        e = np.array([[0.5+0.0j, 0.1+0.0j],
                      [0.0+0.0j, 0.3+0.0j]], dtype=complex, order='F')

        c = np.array([[1.0+1.0j, 2.0+0.5j],
                      [0.5+0.2j, 1.5+1.0j]], dtype=complex, order='F')
        f = np.array([[0.5+0.5j, 1.0+0.3j],
                      [0.3+0.1j, 0.8+0.6j]], dtype=complex, order='F')
        c_orig = c.copy()
        f_orig = f.copy()

        r, l, scale, info = mb04rv(m, n, pmax, a, b, c, d, e, f)

        assert info == 0
        assert 0.0 < scale <= 1.0

        residual1 = a @ r - l @ b - scale * c_orig
        residual2 = d @ r - l @ e - scale * f_orig
        assert_allclose(residual1, 0.0, atol=1e-13)
        assert_allclose(residual2, 0.0, atol=1e-13)

    def test_rectangular_3x2(self):
        """
        Test 3x2 rectangular system.

        Random seed: 456 (for reproducibility)
        """
        from ctrlsys import mb04rv

        np.random.seed(456)

        m, n = 3, 2
        pmax = 1e6

        a = np.array([[2.0+0.5j, 0.3+0.1j, 0.1+0.05j],
                      [0.0+0.0j, 1.5+0.3j, 0.2+0.1j],
                      [0.0+0.0j, 0.0+0.0j, 1.0+0.2j]], dtype=complex, order='F')
        b = np.array([[1.0+0.2j, 0.2+0.1j],
                      [0.0+0.0j, 0.8+0.1j]], dtype=complex, order='F')
        d = np.array([[1.0+0.0j, 0.1+0.05j, 0.05+0.02j],
                      [0.0+0.0j, 0.5+0.0j, 0.1+0.05j],
                      [0.0+0.0j, 0.0+0.0j, 0.3+0.0j]], dtype=complex, order='F')
        e = np.array([[0.5+0.0j, 0.1+0.0j],
                      [0.0+0.0j, 0.4+0.0j]], dtype=complex, order='F')

        c = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        c = np.asfortranarray(c)
        f = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        f = np.asfortranarray(f)
        c_orig = c.copy()
        f_orig = f.copy()

        r, l, scale, info = mb04rv(m, n, pmax, a, b, c, d, e, f)

        assert info == 0
        assert 0.0 < scale <= 1.0

        residual1 = a @ r - l @ b - scale * c_orig
        residual2 = d @ r - l @ e - scale * f_orig
        assert_allclose(residual1, 0.0, atol=1e-12)
        assert_allclose(residual2, 0.0, atol=1e-12)

    def test_rectangular_2x3(self):
        """
        Test 2x3 rectangular system.

        Random seed: 789 (for reproducibility)
        """
        from ctrlsys import mb04rv

        np.random.seed(789)

        m, n = 2, 3
        pmax = 1e6

        a = np.array([[2.0+0.5j, 0.3+0.1j],
                      [0.0+0.0j, 1.5+0.3j]], dtype=complex, order='F')
        b = np.array([[1.0+0.2j, 0.2+0.1j, 0.1+0.05j],
                      [0.0+0.0j, 0.8+0.1j, 0.15+0.05j],
                      [0.0+0.0j, 0.0+0.0j, 0.6+0.1j]], dtype=complex, order='F')
        d = np.array([[1.0+0.0j, 0.1+0.05j],
                      [0.0+0.0j, 0.5+0.0j]], dtype=complex, order='F')
        e = np.array([[0.5+0.0j, 0.1+0.0j, 0.05+0.0j],
                      [0.0+0.0j, 0.4+0.0j, 0.08+0.0j],
                      [0.0+0.0j, 0.0+0.0j, 0.3+0.0j]], dtype=complex, order='F')

        c = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        c = np.asfortranarray(c)
        f = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        f = np.asfortranarray(f)
        c_orig = c.copy()
        f_orig = f.copy()

        r, l, scale, info = mb04rv(m, n, pmax, a, b, c, d, e, f)

        assert info == 0
        assert 0.0 < scale <= 1.0

        residual1 = a @ r - l @ b - scale * c_orig
        residual2 = d @ r - l @ e - scale * f_orig
        assert_allclose(residual1, 0.0, atol=1e-12)
        assert_allclose(residual2, 0.0, atol=1e-12)

    def test_pmax_threshold(self):
        """
        Test that INFO=1 is returned when solution exceeds PMAX.

        Use very small PMAX to trigger the threshold.
        Eigenvalues must be distinct (A/D != B/E) to avoid INFO=2.
        """
        from ctrlsys import mb04rv

        np.random.seed(111)

        m, n = 1, 1
        pmax = 0.001  # Very small threshold

        a = np.array([[2.0 + 1.0j]], dtype=complex, order='F')
        b = np.array([[3.0 + 0.5j]], dtype=complex, order='F')
        d = np.array([[1.0 + 0.0j]], dtype=complex, order='F')
        e = np.array([[1.0 + 0.0j]], dtype=complex, order='F')

        c = np.array([[100.0 + 50.0j]], dtype=complex, order='F')
        f = np.array([[50.0 + 25.0j]], dtype=complex, order='F')

        r, l, scale, info = mb04rv(m, n, pmax, a, b, c, d, e, f)

        assert info == 1

    def test_zero_dimensions(self):
        """
        Test quick return for zero dimensions.
        """
        from ctrlsys import mb04rv

        m, n = 0, 0
        pmax = 1e6

        a = np.array([], dtype=complex, order='F').reshape(0, 0)
        b = np.array([], dtype=complex, order='F').reshape(0, 0)
        c = np.array([], dtype=complex, order='F').reshape(0, 0)
        d = np.array([], dtype=complex, order='F').reshape(0, 0)
        e = np.array([], dtype=complex, order='F').reshape(0, 0)
        f = np.array([], dtype=complex, order='F').reshape(0, 0)

        r, l, scale, info = mb04rv(m, n, pmax, a, b, c, d, e, f)

        assert info == 0
        assert scale == 1.0

    def test_m_zero_n_nonzero(self):
        """
        Test M=0, N>0 case.
        """
        from ctrlsys import mb04rv

        m, n = 0, 2
        pmax = 1e6

        a = np.array([], dtype=complex, order='F').reshape(0, 0)
        b = np.array([[1.0+0.1j, 0.2+0.05j],
                      [0.0+0.0j, 0.8+0.1j]], dtype=complex, order='F')
        c = np.array([], dtype=complex, order='F').reshape(0, 2)
        d = np.array([], dtype=complex, order='F').reshape(0, 0)
        e = np.array([[0.5+0.0j, 0.1+0.0j],
                      [0.0+0.0j, 0.3+0.0j]], dtype=complex, order='F')
        f = np.array([], dtype=complex, order='F').reshape(0, 2)

        r, l, scale, info = mb04rv(m, n, pmax, a, b, c, d, e, f)

        assert info == 0
        assert scale == 1.0

    def test_scale_factor_applied(self):
        """
        Verify that scale factor is properly applied to solution.

        When scale < 1, the returned R and L satisfy:
        A * R - L * B = scale * C_original
        D * R - L * E = scale * F_original

        Random seed: 222 (for reproducibility)
        """
        from ctrlsys import mb04rv

        np.random.seed(222)

        m, n = 2, 2
        pmax = 1e6

        a = np.array([[2.0+1.0j, 0.5+0.2j],
                      [0.0+0.0j, 1.0+0.5j]], dtype=complex, order='F')
        b = np.array([[1.0+0.3j, 0.3+0.1j],
                      [0.0+0.0j, 0.8+0.2j]], dtype=complex, order='F')
        d = np.array([[1.0+0.0j, 0.2+0.1j],
                      [0.0+0.0j, 0.5+0.0j]], dtype=complex, order='F')
        e = np.array([[0.5+0.0j, 0.1+0.0j],
                      [0.0+0.0j, 0.3+0.0j]], dtype=complex, order='F')

        c = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        c = np.asfortranarray(c)
        f = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        f = np.asfortranarray(f)
        c_orig = c.copy()
        f_orig = f.copy()

        r, l, scale, info = mb04rv(m, n, pmax, a, b, c, d, e, f)

        assert info == 0

        lhs1 = a @ r - l @ b
        rhs1 = scale * c_orig
        lhs2 = d @ r - l @ e
        rhs2 = scale * f_orig

        assert_allclose(lhs1, rhs1, atol=1e-13)
        assert_allclose(lhs2, rhs2, atol=1e-13)

    def test_diagonal_only_matrices(self):
        """
        Test with purely diagonal matrices (simplest upper triangular).

        Random seed: 333 (for reproducibility)
        """
        from ctrlsys import mb04rv

        np.random.seed(333)

        m, n = 3, 2
        pmax = 1e6

        a = np.diag([2.0+0.5j, 1.5+0.3j, 1.0+0.2j]).astype(complex, order='F')
        b = np.diag([1.0+0.2j, 0.8+0.1j]).astype(complex, order='F')
        d = np.diag([1.0, 0.5, 0.3]).astype(complex, order='F')
        e = np.diag([0.5, 0.4]).astype(complex, order='F')

        c = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        c = np.asfortranarray(c)
        f = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        f = np.asfortranarray(f)
        c_orig = c.copy()
        f_orig = f.copy()

        r, l, scale, info = mb04rv(m, n, pmax, a, b, c, d, e, f)

        assert info == 0
        assert 0.0 < scale <= 1.0

        residual1 = a @ r - l @ b - scale * c_orig
        residual2 = d @ r - l @ e - scale * f_orig
        assert_allclose(residual1, 0.0, atol=1e-13)
        assert_allclose(residual2, 0.0, atol=1e-13)

    def test_cabs1_property(self):
        """
        Verify understanding of cabs1 (taxi-cab norm).

        The SLICOT implementation uses |real| + |imag| for efficiency.
        """
        z1 = 3.0 + 4.0j
        z2 = -2.0 - 1.0j

        assert cabs1(z1) == 7.0  # |3| + |4|
        assert cabs1(z2) == 3.0  # |-2| + |-1|
        assert cabs1(np.array([z1, z2])).tolist() == [7.0, 3.0]
