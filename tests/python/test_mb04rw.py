"""
Tests for mb04rw - Blocked complex generalized Sylvester equation solver

Solves (using Level 3 BLAS):
    A * R - L * B = scale * C
    D * R - L * E = scale * F

where A, B, D, E are upper triangular complex matrices (generalized Schur form).
Solution (R, L) overwrites (C, F).

This is the blocked version that calls MB04RV for subproblems.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


class TestMB04RW:
    """Tests for mb04rw blocked complex generalized Sylvester solver."""

    def test_basic_2x2(self):
        """
        Test 2x2 system (small case uses MB04RV directly).

        Random seed: 42 (for reproducibility)
        """
        from ctrlsys import mb04rw

        np.random.seed(42)

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

        r, l, scale, info = mb04rw(m, n, pmax, a, b, c, d, e, f)

        assert info == 0
        assert 0.0 < scale <= 1.0

        residual1 = a @ r - l @ b - scale * c_orig
        residual2 = d @ r - l @ e - scale * f_orig
        assert_allclose(residual1, 0.0, atol=1e-13)
        assert_allclose(residual2, 0.0, atol=1e-13)

    def test_larger_system_8x8(self):
        """
        Test 8x8 system to potentially trigger blocked algorithm.

        Random seed: 123 (for reproducibility)
        """
        from ctrlsys import mb04rw

        np.random.seed(123)

        m, n = 8, 8
        pmax = 1e6

        a_diag = np.array([2.0+0.5j, 1.8+0.4j, 1.6+0.3j, 1.4+0.2j,
                          1.2+0.1j, 1.0+0.05j, 0.8+0.02j, 0.6+0.01j])
        a = np.diag(a_diag).astype(complex, order='F')
        for i in range(m-1):
            a[i, i+1] = 0.1 + 0.05j

        b_diag = np.array([1.0+0.2j, 0.9+0.18j, 0.8+0.16j, 0.7+0.14j,
                          0.6+0.12j, 0.5+0.1j, 0.4+0.08j, 0.3+0.06j])
        b = np.diag(b_diag).astype(complex, order='F')
        for i in range(n-1):
            b[i, i+1] = 0.05 + 0.02j

        d = np.diag(np.linspace(1.0, 0.3, m)).astype(complex, order='F')
        for i in range(m-1):
            d[i, i+1] = 0.02 + 0.01j

        e = np.diag(np.linspace(0.5, 0.15, n)).astype(complex, order='F')
        for i in range(n-1):
            e[i, i+1] = 0.01 + 0.005j

        c = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        c = np.asfortranarray(c)
        f = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        f = np.asfortranarray(f)
        c_orig = c.copy()
        f_orig = f.copy()

        r, l, scale, info = mb04rw(m, n, pmax, a, b, c, d, e, f)

        assert info == 0
        assert 0.0 < scale <= 1.0

        residual1 = a @ r - l @ b - scale * c_orig
        residual2 = d @ r - l @ e - scale * f_orig
        assert_allclose(residual1, 0.0, atol=1e-10)
        assert_allclose(residual2, 0.0, atol=1e-10)

    def test_rectangular_10x6(self):
        """
        Test 10x6 rectangular system.

        Random seed: 456 (for reproducibility)
        """
        from ctrlsys import mb04rw

        np.random.seed(456)

        m, n = 10, 6
        pmax = 1e6

        a_diag = np.linspace(2.0+0.5j, 0.5+0.1j, m)
        a = np.diag(a_diag).astype(complex, order='F')
        for i in range(m-1):
            a[i, i+1] = 0.1 + 0.05j

        b_diag = np.linspace(1.0+0.2j, 0.3+0.05j, n)
        b = np.diag(b_diag).astype(complex, order='F')
        for i in range(n-1):
            b[i, i+1] = 0.05 + 0.02j

        d = np.diag(np.linspace(1.0, 0.2, m)).astype(complex, order='F')
        for i in range(m-1):
            d[i, i+1] = 0.02

        e = np.diag(np.linspace(0.5, 0.1, n)).astype(complex, order='F')
        for i in range(n-1):
            e[i, i+1] = 0.01

        c = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        c = np.asfortranarray(c)
        f = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        f = np.asfortranarray(f)
        c_orig = c.copy()
        f_orig = f.copy()

        r, l, scale, info = mb04rw(m, n, pmax, a, b, c, d, e, f)

        assert info == 0
        assert 0.0 < scale <= 1.0

        residual1 = a @ r - l @ b - scale * c_orig
        residual2 = d @ r - l @ e - scale * f_orig
        assert_allclose(residual1, 0.0, atol=1e-11)
        assert_allclose(residual2, 0.0, atol=1e-11)

    def test_rectangular_6x10(self):
        """
        Test 6x10 rectangular system.

        Random seed: 789 (for reproducibility)
        """
        from ctrlsys import mb04rw

        np.random.seed(789)

        m, n = 6, 10
        pmax = 1e6

        a_diag = np.linspace(2.0+0.5j, 0.8+0.2j, m)
        a = np.diag(a_diag).astype(complex, order='F')
        for i in range(m-1):
            a[i, i+1] = 0.1 + 0.05j

        b_diag = np.linspace(1.0+0.2j, 0.2+0.04j, n)
        b = np.diag(b_diag).astype(complex, order='F')
        for i in range(n-1):
            b[i, i+1] = 0.05 + 0.02j

        d = np.diag(np.linspace(1.0, 0.3, m)).astype(complex, order='F')
        for i in range(m-1):
            d[i, i+1] = 0.02

        e = np.diag(np.linspace(0.5, 0.08, n)).astype(complex, order='F')
        for i in range(n-1):
            e[i, i+1] = 0.01

        c = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        c = np.asfortranarray(c)
        f = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        f = np.asfortranarray(f)
        c_orig = c.copy()
        f_orig = f.copy()

        r, l, scale, info = mb04rw(m, n, pmax, a, b, c, d, e, f)

        assert info == 0
        assert 0.0 < scale <= 1.0

        residual1 = a @ r - l @ b - scale * c_orig
        residual2 = d @ r - l @ e - scale * f_orig
        assert_allclose(residual1, 0.0, atol=1e-11)
        assert_allclose(residual2, 0.0, atol=1e-11)

    def test_zero_dimensions(self):
        """
        Test quick return for zero dimensions.
        """
        from ctrlsys import mb04rw

        m, n = 0, 0
        pmax = 1e6

        a = np.array([], dtype=complex, order='F').reshape(0, 0)
        b = np.array([], dtype=complex, order='F').reshape(0, 0)
        c = np.array([], dtype=complex, order='F').reshape(0, 0)
        d = np.array([], dtype=complex, order='F').reshape(0, 0)
        e = np.array([], dtype=complex, order='F').reshape(0, 0)
        f = np.array([], dtype=complex, order='F').reshape(0, 0)

        r, l, scale, info = mb04rw(m, n, pmax, a, b, c, d, e, f)

        assert info == 0
        assert scale == 1.0

    def test_m_zero_n_nonzero(self):
        """
        Test M=0, N>0 case.
        """
        from ctrlsys import mb04rw

        m, n = 0, 3
        pmax = 1e6

        a = np.array([], dtype=complex, order='F').reshape(0, 0)
        b = np.array([[1.0+0.1j, 0.2+0.05j, 0.1+0.02j],
                      [0.0+0.0j, 0.8+0.1j, 0.15+0.03j],
                      [0.0+0.0j, 0.0+0.0j, 0.6+0.05j]], dtype=complex, order='F')
        c = np.array([], dtype=complex, order='F').reshape(0, 3)
        d = np.array([], dtype=complex, order='F').reshape(0, 0)
        e = np.array([[0.5+0.0j, 0.1+0.0j, 0.05+0.0j],
                      [0.0+0.0j, 0.3+0.0j, 0.08+0.0j],
                      [0.0+0.0j, 0.0+0.0j, 0.2+0.0j]], dtype=complex, order='F')
        f = np.array([], dtype=complex, order='F').reshape(0, 3)

        r, l, scale, info = mb04rw(m, n, pmax, a, b, c, d, e, f)

        assert info == 0
        assert scale == 1.0

    def test_pmax_threshold(self):
        """
        Test that INFO=1 is returned when solution exceeds PMAX.
        """
        from ctrlsys import mb04rw

        m, n = 2, 2
        pmax = 0.001  # Very small threshold

        a = np.array([[2.0+1.0j, 0.5+0.2j],
                      [0.0+0.0j, 1.0+0.5j]], dtype=complex, order='F')
        b = np.array([[1.0+0.3j, 0.3+0.1j],
                      [0.0+0.0j, 0.8+0.2j]], dtype=complex, order='F')
        d = np.array([[1.0+0.0j, 0.2+0.1j],
                      [0.0+0.0j, 0.5+0.0j]], dtype=complex, order='F')
        e = np.array([[0.5+0.0j, 0.1+0.0j],
                      [0.0+0.0j, 0.3+0.0j]], dtype=complex, order='F')

        c = np.array([[100.0+50.0j, 200.0+100.0j],
                      [50.0+25.0j, 150.0+75.0j]], dtype=complex, order='F')
        f = np.array([[50.0+25.0j, 100.0+50.0j],
                      [25.0+12.5j, 75.0+37.5j]], dtype=complex, order='F')

        r, l, scale, info = mb04rw(m, n, pmax, a, b, c, d, e, f)

        assert info == 1

    def test_consistency_with_mb04rv(self):
        """
        Verify mb04rw gives same results as mb04rv for small systems.

        Random seed: 555 (for reproducibility)
        """
        from ctrlsys import mb04rw, mb04rv

        np.random.seed(555)

        m, n = 3, 3
        pmax = 1e6

        a = np.array([[2.0+1.0j, 0.5+0.2j, 0.1+0.05j],
                      [0.0+0.0j, 1.5+0.5j, 0.3+0.1j],
                      [0.0+0.0j, 0.0+0.0j, 1.0+0.3j]], dtype=complex, order='F')
        b = np.array([[1.0+0.3j, 0.3+0.1j, 0.1+0.05j],
                      [0.0+0.0j, 0.8+0.2j, 0.2+0.05j],
                      [0.0+0.0j, 0.0+0.0j, 0.5+0.1j]], dtype=complex, order='F')
        d = np.array([[1.0+0.0j, 0.2+0.1j, 0.1+0.05j],
                      [0.0+0.0j, 0.5+0.0j, 0.15+0.05j],
                      [0.0+0.0j, 0.0+0.0j, 0.3+0.0j]], dtype=complex, order='F')
        e = np.array([[0.5+0.0j, 0.1+0.0j, 0.05+0.0j],
                      [0.0+0.0j, 0.3+0.0j, 0.08+0.0j],
                      [0.0+0.0j, 0.0+0.0j, 0.2+0.0j]], dtype=complex, order='F')

        c = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        c = np.asfortranarray(c)
        f = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        f = np.asfortranarray(f)

        c_rw = c.copy()
        f_rw = f.copy()
        c_rv = c.copy()
        f_rv = f.copy()

        r_rw, l_rw, scale_rw, info_rw = mb04rw(m, n, pmax, a.copy(), b.copy(),
                                                c_rw, d.copy(), e.copy(), f_rw)
        r_rv, l_rv, scale_rv, info_rv = mb04rv(m, n, pmax, a.copy(), b.copy(),
                                                c_rv, d.copy(), e.copy(), f_rv)

        assert info_rw == info_rv == 0
        assert_allclose(scale_rw, scale_rv, rtol=1e-14)
        assert_allclose(r_rw, r_rv, rtol=1e-13)
        assert_allclose(l_rw, l_rv, rtol=1e-13)

    def test_sylvester_equation_property(self):
        """
        Test mathematical property: verify Sylvester equations hold.

        For solution (R, L):
          A * R - L * B = scale * C
          D * R - L * E = scale * F

        Random seed: 666 (for reproducibility)
        """
        from ctrlsys import mb04rw

        np.random.seed(666)

        m, n = 5, 4
        pmax = 1e6

        a_diag = np.linspace(2.0+0.5j, 1.0+0.2j, m)
        a = np.diag(a_diag).astype(complex, order='F')
        for i in range(m-1):
            a[i, i+1] = 0.1 + 0.05j

        b_diag = np.linspace(1.0+0.2j, 0.4+0.1j, n)
        b = np.diag(b_diag).astype(complex, order='F')
        for i in range(n-1):
            b[i, i+1] = 0.05 + 0.02j

        d = np.diag(np.linspace(1.0, 0.3, m)).astype(complex, order='F')
        for i in range(m-1):
            d[i, i+1] = 0.02

        e = np.diag(np.linspace(0.5, 0.15, n)).astype(complex, order='F')
        for i in range(n-1):
            e[i, i+1] = 0.01

        c = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        c = np.asfortranarray(c)
        f = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        f = np.asfortranarray(f)
        c_orig = c.copy()
        f_orig = f.copy()

        r, l, scale, info = mb04rw(m, n, pmax, a, b, c, d, e, f)

        assert info == 0

        lhs1 = a @ r - l @ b
        rhs1 = scale * c_orig

        lhs2 = d @ r - l @ e
        rhs2 = scale * f_orig

        assert_allclose(lhs1, rhs1, rtol=1e-13, atol=1e-14)
        assert_allclose(lhs2, rhs2, rtol=1e-13, atol=1e-14)
