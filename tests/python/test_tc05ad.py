"""
Tests for TC05AD - Frequency response of polynomial matrix representation.

TC05AD evaluates the transfer matrix T(s) of a left polynomial matrix
representation [T(s) = inv(P(s))*Q(s)] or a right polynomial matrix
representation [T(s) = Q(s)*inv(P(s))] at a specified complex frequency s = SVAL.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ctrlsys import tc05ad


class TestTC05ADBasic:
    """Basic functionality tests from HTML documentation example."""

    def test_left_pmr_basic(self):
        """
        Test left polynomial matrix representation evaluation.

        From TC05AD HTML documentation example:
        M=2, P=2, SVAL=(0, 0.5j), LERI='L'

        Expected output:
        RCOND = 0.19
        T(SVAL) = [(-0.25,-0.33), (0.26,-0.45)]
                  [(-1.48, 0.35), (-2.25,-1.11)]
        """
        m = 2
        p = 2
        sval = complex(0.0, 0.5)

        index = np.array([2, 2], dtype=np.int32)

        pcoeff = np.zeros((2, 2, 3), dtype=np.float64, order='F')
        pcoeff[0, 0, :] = [2.0, 3.0, 1.0]
        pcoeff[0, 1, :] = [4.0, -1.0, -1.0]
        pcoeff[1, 0, :] = [5.0, 7.0, -6.0]
        pcoeff[1, 1, :] = [3.0, 2.0, 2.0]

        qcoeff = np.zeros((2, 2, 3), dtype=np.float64, order='F')
        qcoeff[0, 0, :] = [6.0, -1.0, 5.0]
        qcoeff[0, 1, :] = [1.0, 7.0, 5.0]
        qcoeff[1, 0, :] = [1.0, 1.0, 1.0]
        qcoeff[1, 1, :] = [4.0, 1.0, -1.0]

        rcond, cfreqr, info = tc05ad('L', m, p, sval, index, pcoeff, qcoeff)

        assert info == 0
        assert rcond == pytest.approx(0.19, abs=0.01)

        expected = np.array([
            [complex(-0.25, -0.33), complex(0.26, -0.45)],
            [complex(-1.48, 0.35), complex(-2.25, -1.11)]
        ], dtype=np.complex128, order='F')

        assert_allclose(cfreqr, expected, rtol=0.01, atol=0.01)


class TestTC05ADMathematicalProperties:
    """Mathematical property validation tests."""

    def test_transfer_function_definition(self):
        """
        Validate T(s) = inv(P(s)) * Q(s) for left PMR.

        For left PMR: T(s) = P(s)^{-1} * Q(s)
        We verify by computing P(s)*T(s) = Q(s).

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)

        m, p = 2, 2
        sval = complex(0.0, 1.0)

        index = np.array([1, 1], dtype=np.int32)
        kpcoef = 2

        pcoeff = np.zeros((p, p, kpcoef), dtype=np.float64, order='F')
        pcoeff[:, :, 0] = np.eye(p)
        pcoeff[:, :, 1] = np.random.randn(p, p) * 0.5

        qcoeff = np.zeros((p, m, kpcoef), dtype=np.float64, order='F')
        qcoeff[:, :, 0] = np.random.randn(p, m) * 0.5
        qcoeff[:, :, 1] = np.random.randn(p, m) * 0.5

        rcond, cfreqr, info = tc05ad('L', m, p, sval, index, pcoeff.copy(), qcoeff.copy())

        assert info == 0
        assert rcond > 0.0

        p_sval = np.zeros((p, p), dtype=np.complex128)
        q_sval = np.zeros((p, m), dtype=np.complex128)

        for i in range(p):
            for j in range(p):
                p_sval[i, j] = pcoeff[i, j, 0]
                for k in range(1, index[i] + 1):
                    p_sval[i, j] = sval * p_sval[i, j] + pcoeff[i, j, k]
            for j in range(m):
                q_sval[i, j] = qcoeff[i, j, 0]
                for k in range(1, index[i] + 1):
                    q_sval[i, j] = sval * q_sval[i, j] + qcoeff[i, j, k]

        product = p_sval @ cfreqr
        assert_allclose(product, q_sval, rtol=1e-12, atol=1e-12)

    def test_right_pmr_square(self):
        """
        Validate T(s) = Q(s) * inv(P(s)) for right PMR (square case).

        For right PMR with m=p (square):
        - P(s) is M x M
        - Q(s) is M x M (stored in max(m,p) x max(m,p) array)
        - T(s) = Q(s) * P(s)^{-1} is M x M

        We verify by computing T(s)*P(s) = Q(s).

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)

        m, p = 2, 2
        sval = complex(0.5, 1.0)

        index = np.array([1, 1], dtype=np.int32)
        kpcoef = 2

        pcoeff = np.zeros((m, m, kpcoef), dtype=np.float64, order='F')
        pcoeff[:, :, 0] = np.eye(m)
        pcoeff[:, :, 1] = np.random.randn(m, m) * 0.3

        qcoeff = np.zeros((m, m, kpcoef), dtype=np.float64, order='F')
        qcoeff[:, :, 0] = np.random.randn(m, m) * 0.5
        qcoeff[:, :, 1] = np.random.randn(m, m) * 0.5

        pcoeff_copy = pcoeff.copy()
        qcoeff_copy = qcoeff.copy()

        rcond, cfreqr, info = tc05ad('R', m, p, sval, index, pcoeff_copy, qcoeff_copy)

        assert info == 0
        assert rcond > 0.0

        p_sval = np.zeros((m, m), dtype=np.complex128)
        q_sval = np.zeros((m, m), dtype=np.complex128)

        for j in range(m):
            for i in range(m):
                p_sval[i, j] = pcoeff[i, j, 0]
                for k in range(1, index[j] + 1):
                    p_sval[i, j] = sval * p_sval[i, j] + pcoeff[i, j, k]
            for i in range(m):
                q_sval[i, j] = qcoeff[i, j, 0]
                for k in range(1, index[j] + 1):
                    q_sval[i, j] = sval * q_sval[i, j] + qcoeff[i, j, k]

        product = cfreqr @ p_sval
        assert_allclose(product, q_sval, rtol=1e-12, atol=1e-12)

    def test_frequency_response_consistency(self):
        """
        Test frequency response at multiple frequencies.

        For a stable system, T(jw) should be continuous as w varies.
        Test that nearby frequencies give nearby values.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)

        m, p = 2, 2
        index = np.array([1, 1], dtype=np.int32)
        kpcoef = 2

        pcoeff = np.zeros((p, p, kpcoef), dtype=np.float64, order='F')
        pcoeff[:, :, 0] = np.eye(p)
        pcoeff[:, :, 1] = 0.1 * np.eye(p)

        qcoeff = np.zeros((p, m, kpcoef), dtype=np.float64, order='F')
        qcoeff[:, :, 0] = 0.5 * np.eye(m)
        qcoeff[:, :, 1] = 0.1 * np.eye(m)

        omega1 = 1.0
        omega2 = 1.001

        _, t1, info1 = tc05ad('L', m, p, complex(0, omega1), index,
                              pcoeff.copy(), qcoeff.copy())
        _, t2, info2 = tc05ad('L', m, p, complex(0, omega2), index,
                              pcoeff.copy(), qcoeff.copy())

        assert info1 == 0
        assert info2 == 0

        rel_diff = np.linalg.norm(t2 - t1) / np.linalg.norm(t1)
        assert rel_diff < 0.01


class TestTC05ADEdgeCases:
    """Edge case tests."""

    def test_scalar_system(self):
        """
        Test with scalar system (m=1, p=1).

        For scalar polynomial P(s) = s + a, Q(s) = b
        T(s) = b / (s + a)
        """
        m, p = 1, 1
        a = 2.0
        b = 3.0
        sval = complex(0.0, 1.0)

        index = np.array([1], dtype=np.int32)

        pcoeff = np.zeros((1, 1, 2), dtype=np.float64, order='F')
        pcoeff[0, 0, 0] = 1.0
        pcoeff[0, 0, 1] = a

        qcoeff = np.zeros((1, 1, 2), dtype=np.float64, order='F')
        qcoeff[0, 0, 0] = 0.0
        qcoeff[0, 0, 1] = b

        rcond, cfreqr, info = tc05ad('L', m, p, sval, index, pcoeff, qcoeff)

        assert info == 0
        assert rcond > 0.0

        expected = b / (sval + a)
        assert_allclose(cfreqr[0, 0], expected, rtol=1e-14)

    def test_zero_frequency(self):
        """
        Test DC gain (s=0).

        For left PMR at s=0: T(0) = P(0)^{-1} * Q(0)
        """
        m, p = 2, 2
        sval = complex(0.0, 0.0)

        index = np.array([1, 1], dtype=np.int32)

        pcoeff = np.zeros((2, 2, 2), dtype=np.float64, order='F')
        pcoeff[0, 0, 0] = 1.0
        pcoeff[0, 0, 1] = 1.0
        pcoeff[1, 1, 0] = 1.0
        pcoeff[1, 1, 1] = 2.0

        qcoeff = np.zeros((2, 2, 2), dtype=np.float64, order='F')
        qcoeff[0, 0, 1] = 3.0
        qcoeff[1, 1, 1] = 4.0

        rcond, cfreqr, info = tc05ad('L', m, p, sval, index, pcoeff, qcoeff)

        assert info == 0

        expected = np.array([
            [3.0, 0.0],
            [0.0, 2.0]
        ], dtype=np.complex128)

        assert_allclose(cfreqr.real, expected.real, rtol=1e-14)
        assert_allclose(cfreqr.imag, expected.imag, atol=1e-14)

    def test_high_degree_polynomial(self):
        """
        Test with higher degree polynomials.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)

        m, p = 2, 2
        max_deg = 4
        sval = complex(0.0, 0.5)

        index = np.array([max_deg, max_deg], dtype=np.int32)
        kpcoef = max_deg + 1

        pcoeff = np.zeros((p, p, kpcoef), dtype=np.float64, order='F')
        pcoeff[:, :, 0] = np.eye(p)
        for k in range(1, kpcoef):
            pcoeff[:, :, k] = np.random.randn(p, p) * (0.5 ** k)

        qcoeff = np.zeros((p, m, kpcoef), dtype=np.float64, order='F')
        for k in range(kpcoef):
            qcoeff[:, :, k] = np.random.randn(p, m) * (0.5 ** k)

        rcond, cfreqr, info = tc05ad('L', m, p, sval, index, pcoeff.copy(), qcoeff.copy())

        assert info == 0
        assert rcond > 0.0
        assert cfreqr.shape == (p, m)


class TestTC05ADErrorHandling:
    """Error handling tests."""

    def test_singular_denominator(self):
        """
        Test behavior when P(SVAL) is singular.

        INFO should be 1 when P(SVAL) is exactly or nearly singular.
        """
        m, p = 2, 2
        sval = complex(0.0, 0.0)

        index = np.array([0, 0], dtype=np.int32)

        pcoeff = np.zeros((2, 2, 1), dtype=np.float64, order='F')
        pcoeff[0, 0, 0] = 1.0
        pcoeff[0, 1, 0] = 2.0
        pcoeff[1, 0, 0] = 2.0
        pcoeff[1, 1, 0] = 4.0

        qcoeff = np.zeros((2, 2, 1), dtype=np.float64, order='F')
        qcoeff[0, 0, 0] = 1.0
        qcoeff[1, 1, 0] = 1.0

        rcond, cfreqr, info = tc05ad('L', m, p, sval, index, pcoeff, qcoeff)

        assert info == 1
        assert rcond < 1e-10

    def test_invalid_leri(self):
        """Test with invalid LERI parameter."""
        m, p = 2, 2
        sval = complex(0.0, 0.5)
        index = np.array([1, 1], dtype=np.int32)
        pcoeff = np.zeros((2, 2, 2), dtype=np.float64, order='F')
        qcoeff = np.zeros((2, 2, 2), dtype=np.float64, order='F')

        with pytest.raises(ValueError):
            tc05ad('X', m, p, sval, index, pcoeff, qcoeff)


class TestTC05ADRightFraction:
    """Tests specifically for right polynomial matrix representation."""

    def test_right_pmr_basic(self):
        """
        Test right polynomial matrix representation evaluation.

        For right PMR in SLICOT:
        - P(s) is M x M
        - Q(s) is M x P (stored in array with ldqco1 >= max(m,p))
        - T(s) = Q(s) * inv(P(s)) is M x P

        Random seed: 999 (for reproducibility)
        """
        np.random.seed(999)

        m, p = 2, 3
        sval = complex(0.0, 1.0)

        index = np.array([1, 1], dtype=np.int32)
        kpcoef = 2

        pcoeff = np.zeros((m, m, kpcoef), dtype=np.float64, order='F')
        pcoeff[:, :, 0] = np.eye(m)
        pcoeff[:, :, 1] = 0.3 * np.random.randn(m, m)

        max_dim = max(m, p)
        qcoeff = np.zeros((max_dim, max_dim, kpcoef), dtype=np.float64, order='F')
        qcoeff[:m, :p, 0] = 0.5 * np.random.randn(m, p)
        qcoeff[:m, :p, 1] = 0.5 * np.random.randn(m, p)

        rcond, cfreqr, info = tc05ad('R', m, p, sval, index, pcoeff, qcoeff)

        assert info == 0
        assert rcond > 0.0
        assert cfreqr.shape == (m, p)

    def test_coefficients_restored_right_square(self):
        """
        Verify PCOEFF and QCOEFF are restored after right PMR evaluation (square case).

        According to docs, PCOEFF and QCOEFF are modified but restored on exit.
        For right PMR with m=p: Q(s) is M x M.
        """
        np.random.seed(111)

        m, p = 2, 2
        sval = complex(0.0, 1.0)

        index = np.array([1, 1], dtype=np.int32)
        kpcoef = 2

        pcoeff = np.zeros((m, m, kpcoef), dtype=np.float64, order='F')
        pcoeff[:, :, 0] = np.eye(m)
        pcoeff[:, :, 1] = 0.3 * np.random.randn(m, m)

        qcoeff = np.zeros((m, m, kpcoef), dtype=np.float64, order='F')
        qcoeff[:, :, 0] = 0.5 * np.random.randn(m, m)
        qcoeff[:, :, 1] = 0.5 * np.random.randn(m, m)

        pcoeff_orig = pcoeff.copy()
        qcoeff_orig = qcoeff.copy()

        _, _, info = tc05ad('R', m, p, sval, index, pcoeff, qcoeff)

        assert info == 0
        assert_allclose(pcoeff, pcoeff_orig, rtol=1e-14)
        assert_allclose(qcoeff, qcoeff_orig, rtol=1e-14)
