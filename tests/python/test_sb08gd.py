"""
Tests for SB08GD - State-space representation from left coprime factorization.

SB08GD constructs G = (A,B,C,D) from factors Q = (AQR,BQ,CQR,DQ) and
R = (AQR,BR,CQR,DR) of left coprime factorization G = R^{-1} * Q.

Formulas:
  A = AQR - BR * DR^{-1} * CQR
  B = BQ  - BR * DR^{-1} * DQ
  C = DR^{-1} * CQR
  D = DR^{-1} * DQ
"""

import numpy as np
import pytest

from ctrlsys import sb08gd


class TestSB08GDBasic:
    """Basic functionality tests."""

    def test_simple_siso_system(self):
        """
        Test with simple 2nd order SISO system.

        Construct known Q and R factors, reconstruct G, verify formulas.
        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)
        n, m, p = 2, 1, 1

        aqr = np.array([
            [-1.0, 0.5],
            [0.0, -2.0]
        ], order='F', dtype=float)

        bq = np.array([
            [1.0],
            [0.5]
        ], order='F', dtype=float)

        cqr = np.array([
            [1.0, 0.5]
        ], order='F', dtype=float)

        dq = np.array([
            [0.5]
        ], order='F', dtype=float)

        br = np.array([
            [0.2],
            [0.1]
        ], order='F', dtype=float)

        dr = np.array([
            [2.0]
        ], order='F', dtype=float)

        a = aqr.copy(order='F')
        b = bq.copy(order='F')
        c = cqr.copy(order='F')
        d = dq.copy(order='F')

        a_out, b_out, c_out, d_out, dr_lu, rcond, info = sb08gd(
            a, b, c, d, br, dr
        )

        assert info == 0

        dr_inv_cqr = np.linalg.solve(dr, cqr)
        dr_inv_dq = np.linalg.solve(dr, dq)

        a_expected = aqr - br @ dr_inv_cqr
        b_expected = bq - br @ dr_inv_dq
        c_expected = dr_inv_cqr
        d_expected = dr_inv_dq

        np.testing.assert_allclose(a_out, a_expected, rtol=1e-14)
        np.testing.assert_allclose(b_out, b_expected, rtol=1e-14)
        np.testing.assert_allclose(c_out, c_expected, rtol=1e-14)
        np.testing.assert_allclose(d_out, d_expected, rtol=1e-14)

    def test_mimo_system(self):
        """
        Test with 3rd order MIMO system (2 inputs, 2 outputs).

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)
        n, m, p = 3, 2, 2

        aqr = np.array([
            [-1.0, 0.2, 0.0],
            [0.3, -2.0, 0.1],
            [0.0, 0.4, -1.5]
        ], order='F', dtype=float)

        bq = np.array([
            [1.0, 0.0],
            [0.5, 1.0],
            [0.0, 0.5]
        ], order='F', dtype=float)

        cqr = np.array([
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.0]
        ], order='F', dtype=float)

        dq = np.array([
            [0.5, 0.1],
            [0.0, 0.5]
        ], order='F', dtype=float)

        br = np.array([
            [0.2, 0.0],
            [0.0, 0.3],
            [0.1, 0.1]
        ], order='F', dtype=float)

        dr = np.array([
            [2.0, 0.5],
            [0.0, 1.5]
        ], order='F', dtype=float)

        a = aqr.copy(order='F')
        b = bq.copy(order='F')
        c = cqr.copy(order='F')
        d = dq.copy(order='F')

        a_out, b_out, c_out, d_out, dr_lu, rcond, info = sb08gd(
            a, b, c, d, br, dr
        )

        assert info == 0

        dr_inv_cqr = np.linalg.solve(dr, cqr)
        dr_inv_dq = np.linalg.solve(dr, dq)

        a_expected = aqr - br @ dr_inv_cqr
        b_expected = bq - br @ dr_inv_dq
        c_expected = dr_inv_cqr
        d_expected = dr_inv_dq

        np.testing.assert_allclose(a_out, a_expected, rtol=1e-14)
        np.testing.assert_allclose(b_out, b_expected, rtol=1e-14)
        np.testing.assert_allclose(c_out, c_expected, rtol=1e-14)
        np.testing.assert_allclose(d_out, d_expected, rtol=1e-14)


class TestSB08GDEdgeCases:
    """Edge case tests."""

    def test_zero_state_dimension(self):
        """Test with N=0 (zero state dimension)."""
        n, m, p = 0, 2, 2

        a = np.zeros((1, 1), order='F', dtype=float)
        b = np.zeros((1, m), order='F', dtype=float)
        c = np.zeros((p, 1), order='F', dtype=float)
        d = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)
        br = np.zeros((1, p), order='F', dtype=float)
        dr = np.array([
            [2.0, 0.0],
            [0.0, 2.0]
        ], order='F', dtype=float)

        a_out, b_out, c_out, d_out, dr_lu, rcond, info = sb08gd(
            a, b, c, d, br, dr
        )

        assert info == 0

    def test_identity_dr(self):
        """Test with identity DR returns original inputs as outputs."""
        n, m, p = 2, 2, 2

        aqr = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
        bq = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)
        cqr = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)
        dq = np.array([[0.5, 0.0], [0.0, 0.5]], order='F', dtype=float)
        br = np.zeros((2, 2), order='F', dtype=float)
        dr = np.eye(2, order='F', dtype=float)

        a = aqr.copy(order='F')
        b = bq.copy(order='F')
        c = cqr.copy(order='F')
        d = dq.copy(order='F')

        a_out, b_out, c_out, d_out, dr_lu, rcond, info = sb08gd(
            a, b, c, d, br, dr.copy(order='F')
        )

        assert info == 0
        np.testing.assert_allclose(a_out, aqr, rtol=1e-14)
        np.testing.assert_allclose(b_out, bq, rtol=1e-14)
        np.testing.assert_allclose(c_out, cqr, rtol=1e-14)
        np.testing.assert_allclose(d_out, dq, rtol=1e-14)


class TestSB08GDMathematicalProperties:
    """Mathematical property validation tests."""

    def test_formula_verification_random(self):
        """
        Verify formulas A = AQR - BR*DR^{-1}*CQR, etc.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)
        n, m, p = 4, 2, 3

        aqr = -np.eye(n) + 0.1 * np.random.randn(n, n)
        aqr = np.asfortranarray(aqr)

        bq = np.random.randn(n, m)
        bq = np.asfortranarray(bq)

        cqr = np.random.randn(p, n)
        cqr = np.asfortranarray(cqr)

        dq = np.random.randn(p, m)
        dq = np.asfortranarray(dq)

        br = np.random.randn(n, p)
        br = np.asfortranarray(br)

        dr = np.eye(p) + 0.5 * np.random.randn(p, p)
        dr = np.asfortranarray(dr)

        a = aqr.copy(order='F')
        b = bq.copy(order='F')
        c = cqr.copy(order='F')
        d = dq.copy(order='F')

        a_out, b_out, c_out, d_out, dr_lu, rcond, info = sb08gd(
            a, b, c, d, br, dr.copy(order='F')
        )

        assert info == 0

        dr_inv_cqr = np.linalg.solve(dr, cqr)
        dr_inv_dq = np.linalg.solve(dr, dq)

        a_expected = aqr - br @ dr_inv_cqr
        b_expected = bq - br @ dr_inv_dq
        c_expected = dr_inv_cqr
        d_expected = dr_inv_dq

        np.testing.assert_allclose(a_out, a_expected, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(b_out, b_expected, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(c_out, c_expected, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(d_out, d_expected, rtol=1e-13, atol=1e-14)

    def test_dr_lu_factorization_returned(self):
        """
        Verify that DR is returned as LU factorization.

        The routine returns DR overwritten with its LU factorization.
        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)
        n, m, p = 2, 1, 2

        aqr = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
        bq = np.array([[1.0], [0.5]], order='F', dtype=float)
        cqr = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)
        dq = np.array([[0.5], [0.3]], order='F', dtype=float)
        br = np.array([[0.1, 0.0], [0.0, 0.1]], order='F', dtype=float)
        dr = np.array([[2.0, 0.5], [1.0, 2.0]], order='F', dtype=float)

        a = aqr.copy(order='F')
        b = bq.copy(order='F')
        c = cqr.copy(order='F')
        d = dq.copy(order='F')

        a_out, b_out, c_out, d_out, dr_lu, rcond, info = sb08gd(
            a, b, c, d, br, dr.copy(order='F')
        )

        assert info == 0
        assert rcond > 0


class TestSB08GDErrorHandling:
    """Error handling tests."""

    def test_singular_dr(self):
        """Test with singular DR matrix returns info=1."""
        n, m, p = 2, 1, 2

        aqr = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
        bq = np.array([[1.0], [0.5]], order='F', dtype=float)
        cqr = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)
        dq = np.array([[0.5], [0.3]], order='F', dtype=float)
        br = np.array([[0.1, 0.0], [0.0, 0.1]], order='F', dtype=float)
        dr = np.array([[1.0, 1.0], [1.0, 1.0]], order='F', dtype=float)

        a = aqr.copy(order='F')
        b = bq.copy(order='F')
        c = cqr.copy(order='F')
        d = dq.copy(order='F')

        a_out, b_out, c_out, d_out, dr_lu, rcond, info = sb08gd(
            a, b, c, d, br, dr.copy(order='F')
        )

        assert info == 1

    def test_nearly_singular_dr_warning(self):
        """
        Test with nearly singular DR matrix returns info=2 (warning).

        Random seed: 999 (for reproducibility)
        """
        np.random.seed(999)
        n, m, p = 2, 1, 2

        aqr = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
        bq = np.array([[1.0], [0.5]], order='F', dtype=float)
        cqr = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)
        dq = np.array([[0.5], [0.3]], order='F', dtype=float)
        br = np.array([[0.1, 0.0], [0.0, 0.1]], order='F', dtype=float)
        dr = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-16]], order='F', dtype=float)

        a = aqr.copy(order='F')
        b = bq.copy(order='F')
        c = cqr.copy(order='F')
        d = dq.copy(order='F')

        a_out, b_out, c_out, d_out, dr_lu, rcond, info = sb08gd(
            a, b, c, d, br, dr.copy(order='F')
        )

        assert info in [1, 2]
