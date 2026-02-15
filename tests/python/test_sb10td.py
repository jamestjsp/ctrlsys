"""
Tests for SB10TD - H2 optimal discrete-time controller transformation.

Transforms controller matrices from normalized system to original system.
"""
import numpy as np
import pytest
import ctrlsys


class TestSB10TDBasic:
    """Basic functionality tests for SB10TD."""

    def test_basic_2x3x3(self):
        """
        Test basic transformation with a 2x3x3 system (n=2, m=3, np=3).

        Uses identity transformation matrices to verify pass-through behavior.
        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)

        n, m, np_ = 2, 3, 3
        ncon, nmeas = 1, 1

        d = np.zeros((np_, m), dtype=np.float64, order='F')
        d[np_ - nmeas:, m - ncon:] = 0.1

        tu = np.eye(ncon, dtype=np.float64, order='F')
        ty = np.eye(nmeas, dtype=np.float64, order='F')

        ak_in = np.array([[0.5, 0.1], [0.0, 0.4]], dtype=np.float64, order='F')
        bk_in = np.array([[0.2], [0.3]], dtype=np.float64, order='F')
        ck_in = np.array([[0.1, 0.2]], dtype=np.float64, order='F')
        dk_in = np.array([[0.05]], dtype=np.float64, order='F')

        ak, bk, ck, dk, rcond, info = ctrlsys.sb10td(
            n, m, np_, ncon, nmeas,
            d, tu, ty, ak_in, bk_in, ck_in, dk_in
        )

        assert info == 0, f"SB10TD returned info = {info}"
        assert ak.shape == (n, n)
        assert bk.shape == (n, nmeas)
        assert ck.shape == (ncon, n)
        assert dk.shape == (ncon, nmeas)
        assert rcond > 0.0, "RCOND should be positive for well-conditioned problem"

    def test_transformation_formula(self):
        """
        Test that transformation follows expected formula.

        With identity TU, TY and zero D22:
        - BKHAT = BK * TY = BK
        - CKHAT = TU * CK = CK
        - DKHAT = TU * DK * TY = DK
        - Since D22 = 0, (I + DKHAT*D22) = I, so CK, DK unchanged
        - AK = AK_in - BK*D22*CK = AK_in when D22 = 0

        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)

        n, m, np_ = 3, 4, 4
        ncon, nmeas = 2, 2

        d = np.zeros((np_, m), dtype=np.float64, order='F')

        tu = np.eye(ncon, dtype=np.float64, order='F')
        ty = np.eye(nmeas, dtype=np.float64, order='F')

        ak_in = np.random.randn(n, n).astype(np.float64, order='F') * 0.1
        bk_in = np.random.randn(n, nmeas).astype(np.float64, order='F') * 0.1
        ck_in = np.random.randn(ncon, n).astype(np.float64, order='F') * 0.1
        dk_in = np.random.randn(ncon, nmeas).astype(np.float64, order='F') * 0.1

        ak, bk, ck, dk, rcond, info = ctrlsys.sb10td(
            n, m, np_, ncon, nmeas,
            d, tu, ty,
            ak_in.copy(order='F'), bk_in.copy(order='F'),
            ck_in.copy(order='F'), dk_in.copy(order='F')
        )

        assert info == 0, f"SB10TD returned info = {info}"

        np.testing.assert_allclose(ak, ak_in, rtol=1e-14, atol=1e-15)
        np.testing.assert_allclose(bk, bk_in, rtol=1e-14, atol=1e-15)
        np.testing.assert_allclose(ck, ck_in, rtol=1e-14, atol=1e-15)
        np.testing.assert_allclose(dk, dk_in, rtol=1e-14, atol=1e-15)


class TestSB10TDNonTrivialTransform:
    """Test cases with non-trivial transformations."""

    def test_nontrivial_tu_ty(self):
        """
        Test with non-identity transformation matrices.

        Verifies correct application of TU and TY transformations.
        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)

        n, m, np_ = 2, 3, 3
        ncon, nmeas = 1, 1

        d = np.zeros((np_, m), dtype=np.float64, order='F')
        d[np_ - nmeas:, m - ncon:] = 0.0

        tu = np.array([[2.0]], dtype=np.float64, order='F')
        ty = np.array([[0.5]], dtype=np.float64, order='F')

        ak_in = np.array([[0.5, 0.1], [0.0, 0.4]], dtype=np.float64, order='F')
        bk_in = np.array([[0.2], [0.3]], dtype=np.float64, order='F')
        ck_in = np.array([[0.1, 0.2]], dtype=np.float64, order='F')
        dk_in = np.array([[0.1]], dtype=np.float64, order='F')

        ak, bk, ck, dk, rcond, info = ctrlsys.sb10td(
            n, m, np_, ncon, nmeas,
            d, tu, ty,
            ak_in.copy(order='F'), bk_in.copy(order='F'),
            ck_in.copy(order='F'), dk_in.copy(order='F')
        )

        assert info == 0, f"SB10TD returned info = {info}"

        bk_expected = bk_in @ ty
        ck_expected = tu @ ck_in
        dk_expected = tu @ dk_in @ ty

        np.testing.assert_allclose(bk, bk_expected, rtol=1e-14, atol=1e-15)
        np.testing.assert_allclose(ck, ck_expected, rtol=1e-14, atol=1e-15)
        np.testing.assert_allclose(dk, dk_expected, rtol=1e-14, atol=1e-15)

    def test_with_nonzero_d22(self):
        """
        Test transformation with nonzero D22 submatrix.

        Verifies the full transformation when feedback loop exists.
        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)

        n, m, np_ = 3, 4, 4
        ncon, nmeas = 2, 2

        d = np.zeros((np_, m), dtype=np.float64, order='F')
        d22 = np.random.randn(nmeas, ncon).astype(np.float64) * 0.1
        d[np_ - nmeas:, m - ncon:] = d22

        tu = np.eye(ncon, dtype=np.float64, order='F')
        ty = np.eye(nmeas, dtype=np.float64, order='F')

        ak_in = np.random.randn(n, n).astype(np.float64, order='F') * 0.2
        bk_in = np.random.randn(n, nmeas).astype(np.float64, order='F') * 0.2
        ck_in = np.random.randn(ncon, n).astype(np.float64, order='F') * 0.2
        dk_in = np.random.randn(ncon, nmeas).astype(np.float64, order='F') * 0.2

        ak, bk, ck, dk, rcond, info = ctrlsys.sb10td(
            n, m, np_, ncon, nmeas,
            d, tu, ty,
            ak_in.copy(order='F'), bk_in.copy(order='F'),
            ck_in.copy(order='F'), dk_in.copy(order='F')
        )

        assert info == 0, f"SB10TD returned info = {info}"
        assert ak.shape == (n, n)
        assert bk.shape == (n, nmeas)
        assert ck.shape == (ncon, n)
        assert dk.shape == (ncon, nmeas)
        assert rcond > 0.0

        bkhat = bk_in @ ty
        ckhat = tu @ ck_in
        dkhat = tu @ dk_in @ ty

        Im2_plus_dkhat_d22 = np.eye(ncon) + dkhat @ d22
        ck_expected = np.linalg.solve(Im2_plus_dkhat_d22, ckhat)
        dk_expected = np.linalg.solve(Im2_plus_dkhat_d22, dkhat)

        temp = bkhat @ d22
        ak_expected = ak_in - temp @ ck_expected
        bk_expected = bkhat - temp @ dk_expected

        np.testing.assert_allclose(ak, ak_expected, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(bk, bk_expected, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(ck, ck_expected, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(dk, dk_expected, rtol=1e-13, atol=1e-14)


class TestSB10TDEdgeCases:
    """Edge case tests for SB10TD."""

    def test_zero_order_system(self):
        """Test with n=0 (zero-order system)."""
        n, m, np_ = 0, 2, 2
        ncon, nmeas = 1, 1

        d = np.zeros((np_, m), dtype=np.float64, order='F')
        tu = np.eye(ncon, dtype=np.float64, order='F')
        ty = np.eye(nmeas, dtype=np.float64, order='F')

        ak_in = np.zeros((1, 0), dtype=np.float64, order='F').reshape(0, 0, order='F')
        bk_in = np.zeros((0, nmeas), dtype=np.float64, order='F')
        ck_in = np.zeros((ncon, 0), dtype=np.float64, order='F')
        dk_in = np.array([[0.1]], dtype=np.float64, order='F')

        ak, bk, ck, dk, rcond, info = ctrlsys.sb10td(
            n, m, np_, ncon, nmeas,
            d, tu, ty, ak_in, bk_in, ck_in, dk_in
        )

        assert info == 0

    def test_single_control_single_measurement(self):
        """
        Test SISO controller (ncon=1, nmeas=1).

        Random seed: 999 (for reproducibility)
        """
        np.random.seed(999)

        n, m, np_ = 4, 3, 3
        ncon, nmeas = 1, 1

        d = np.zeros((np_, m), dtype=np.float64, order='F')
        d[np_ - nmeas:, m - ncon:] = 0.05

        tu = np.array([[1.5]], dtype=np.float64, order='F')
        ty = np.array([[0.8]], dtype=np.float64, order='F')

        ak_in = np.random.randn(n, n).astype(np.float64, order='F') * 0.1
        bk_in = np.random.randn(n, nmeas).astype(np.float64, order='F') * 0.1
        ck_in = np.random.randn(ncon, n).astype(np.float64, order='F') * 0.1
        dk_in = np.random.randn(ncon, nmeas).astype(np.float64, order='F') * 0.1

        ak, bk, ck, dk, rcond, info = ctrlsys.sb10td(
            n, m, np_, ncon, nmeas,
            d, tu, ty,
            ak_in.copy(order='F'), bk_in.copy(order='F'),
            ck_in.copy(order='F'), dk_in.copy(order='F')
        )

        assert info == 0
        assert ak.shape == (n, n)
        assert bk.shape == (n, nmeas)
        assert ck.shape == (ncon, n)
        assert dk.shape == (ncon, nmeas)


class TestSB10TDMIMO:
    """MIMO tests for SB10TD."""

    def test_mimo_ncon2_nmeas2(self):
        """Test MIMO transformation with ncon=2, nmeas=2."""
        np.random.seed(333)

        n, m, np_ = 4, 6, 6
        ncon, nmeas = 2, 2

        d = np.zeros((np_, m), dtype=np.float64, order='F')
        d22 = np.random.randn(nmeas, ncon).astype(np.float64) * 0.05
        d[np_ - nmeas:, m - ncon:] = d22

        tu = np.eye(ncon, dtype=np.float64, order='F') * 1.5
        ty = np.eye(nmeas, dtype=np.float64, order='F') * 0.8

        ak_in = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(np.float64, order='F')
        bk_in = np.random.randn(n, nmeas).astype(np.float64, order='F') * 0.2
        ck_in = np.random.randn(ncon, n).astype(np.float64, order='F') * 0.2
        dk_in = np.random.randn(ncon, nmeas).astype(np.float64, order='F') * 0.1

        ak, bk, ck, dk, rcond, info = ctrlsys.sb10td(
            n, m, np_, ncon, nmeas,
            d, tu, ty,
            ak_in.copy(order='F'), bk_in.copy(order='F'),
            ck_in.copy(order='F'), dk_in.copy(order='F')
        )

        assert info == 0
        assert ak.shape == (n, n)
        assert bk.shape == (n, nmeas)
        assert ck.shape == (ncon, n)
        assert dk.shape == (ncon, nmeas)
        assert rcond > 0.0

        bkhat = bk_in @ ty
        ckhat = tu @ ck_in
        dkhat = tu @ dk_in @ ty

        Im2_plus_dkhat_d22 = np.eye(ncon) + dkhat @ d22
        ck_expected = np.linalg.solve(Im2_plus_dkhat_d22, ckhat)
        dk_expected = np.linalg.solve(Im2_plus_dkhat_d22, dkhat)

        temp = bkhat @ d22
        ak_expected = ak_in - temp @ ck_expected
        bk_expected = bkhat - temp @ dk_expected

        np.testing.assert_allclose(ak, ak_expected, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(bk, bk_expected, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(ck, ck_expected, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(dk, dk_expected, rtol=1e-12, atol=1e-13)


class TestSB10TDErrorHandling:
    """Error handling tests for SB10TD."""

    def test_singular_matrix_error(self):
        """
        Test that info=1 is returned when (I + DKHAT*D22) is singular.

        With TU=TY=I, DKHAT=DK. For I + DK*D22 = 0, use DK=-1, D22=1.
        """
        n, m, np_ = 2, 2, 2
        ncon, nmeas = 1, 1

        d = np.zeros((np_, m), dtype=np.float64, order='F')
        d[np_ - nmeas:, m - ncon:] = 1.0

        tu = np.eye(ncon, dtype=np.float64, order='F')
        ty = np.eye(nmeas, dtype=np.float64, order='F')

        ak_in = np.array([[0.5, 0.1], [0.0, 0.4]], dtype=np.float64, order='F')
        bk_in = np.array([[0.2], [0.3]], dtype=np.float64, order='F')
        ck_in = np.array([[0.1, 0.2]], dtype=np.float64, order='F')
        dk_in = np.array([[-1.0]], dtype=np.float64, order='F')

        ak, bk, ck, dk, rcond, info = ctrlsys.sb10td(
            n, m, np_, ncon, nmeas,
            d, tu, ty,
            ak_in.copy(order='F'), bk_in.copy(order='F'),
            ck_in.copy(order='F'), dk_in.copy(order='F')
        )

        assert info == 1, f"Expected info=1 for singular matrix, got {info}"
