"""
Tests for SLICOT SB10WD routine.

SB10WD computes the H2 optimal controller matrices (AK, BK, CK, DK) from
state feedback matrix F and output injection matrix H.

Controller formulas:
    AK = A + H*C2 + B2*F + H*D22*F
    BK = -H*TY
    CK = TU*F
    DK = 0

where:
    B2 = B(:, M-M2+1:M)      (last M2 columns of B)
    C2 = C(NP-NP2+1:NP, :)   (last NP2 rows of C)
    D22 = D(NP-NP2+1:NP, M-M2+1:M)
"""

import numpy as np
import pytest
from ctrlsys import sb10wd


class TestSB10WDBasic:
    """Basic functionality tests for SB10WD."""

    def test_basic_2x2_system(self):
        """
        Test basic 2x2 system with 1 control input and 1 measurement.

        Random seed: 42 (for reproducibility)
        """
        np.random.seed(42)

        n = 2      # State order
        m = 2      # Total inputs (M1 + M2)
        np_ = 2    # Total outputs (NP1 + NP2)
        ncon = 1   # Control inputs (M2)
        nmeas = 1  # Measurements (NP2)

        m1 = m - ncon      # = 1
        m2 = ncon          # = 1
        np1 = np_ - nmeas  # = 1
        np2 = nmeas        # = 1

        A = np.array([[-1.0, 0.5],
                      [0.0, -2.0]], order='F', dtype=float)

        B = np.array([[1.0, 0.5],
                      [0.0, 1.0]], order='F', dtype=float)

        C = np.array([[1.0, 0.0],
                      [0.5, 1.0]], order='F', dtype=float)

        D = np.array([[0.0, 0.1],
                      [0.0, 0.2]], order='F', dtype=float)

        F = np.array([[-0.5, -0.3]], order='F', dtype=float)

        H = np.array([[0.4],
                      [0.6]], order='F', dtype=float)

        TU = np.array([[1.0]], order='F', dtype=float)
        TY = np.array([[1.0]], order='F', dtype=float)

        B2 = B[:, m1:m]
        C2 = C[np1:np_, :]
        D22 = D[np1:np_, m1:m]

        AK_expected = A + H @ C2 + B2 @ F + H @ D22 @ F
        BK_expected = -H @ TY
        CK_expected = TU @ F
        DK_expected = np.zeros((m2, np2), order='F', dtype=float)

        AK, BK, CK, DK, info = sb10wd(
            n, m, np_, ncon, nmeas,
            A, B, C, D, F, H, TU, TY
        )

        assert info == 0
        np.testing.assert_allclose(AK, AK_expected, rtol=1e-14)
        np.testing.assert_allclose(BK, BK_expected, rtol=1e-14)
        np.testing.assert_allclose(CK, CK_expected, rtol=1e-14)
        np.testing.assert_allclose(DK, DK_expected, rtol=1e-14)

    def test_larger_system(self):
        """
        Test 4x4 system with multiple control inputs and measurements.

        N=4, M=4, NP=4, NCON=2, NMEAS=2
        Random seed: 123 (for reproducibility)
        """
        np.random.seed(123)

        n = 4
        m = 4
        np_ = 4
        ncon = 2
        nmeas = 2

        m1 = m - ncon
        m2 = ncon
        np1 = np_ - nmeas
        np2 = nmeas

        A = np.random.randn(n, n).astype(float, order='F')
        B = np.random.randn(n, m).astype(float, order='F')
        C = np.random.randn(np_, n).astype(float, order='F')
        D = np.random.randn(np_, m).astype(float, order='F')
        F = np.random.randn(m2, n).astype(float, order='F')
        H = np.random.randn(n, np2).astype(float, order='F')
        TU = np.random.randn(m2, m2).astype(float, order='F')
        TY = np.random.randn(np2, np2).astype(float, order='F')

        B2 = B[:, m1:m]
        C2 = C[np1:np_, :]
        D22 = D[np1:np_, m1:m]

        AK_expected = A + H @ C2 + B2 @ F + H @ D22 @ F
        BK_expected = -H @ TY
        CK_expected = TU @ F
        DK_expected = np.zeros((m2, np2), order='F', dtype=float)

        AK, BK, CK, DK, info = sb10wd(
            n, m, np_, ncon, nmeas,
            A, B, C, D, F, H, TU, TY
        )

        assert info == 0
        np.testing.assert_allclose(AK, AK_expected, rtol=1e-14)
        np.testing.assert_allclose(BK, BK_expected, rtol=1e-14)
        np.testing.assert_allclose(CK, CK_expected, rtol=1e-14)
        np.testing.assert_allclose(DK, DK_expected, rtol=1e-14)


class TestSB10WDEdgeCases:
    """Edge case tests for SB10WD."""

    def test_n_zero(self):
        """Test quick return when n=0."""
        n = 0
        m = 2
        np_ = 2
        ncon = 1
        nmeas = 1

        A = np.array([[]], order='F', dtype=float).reshape(0, 0)
        B = np.array([[]], order='F', dtype=float).reshape(0, m)
        C = np.array([[]], order='F', dtype=float).reshape(np_, 0)
        D = np.zeros((np_, m), order='F', dtype=float)
        F = np.array([[]], order='F', dtype=float).reshape(ncon, 0)
        H = np.array([[]], order='F', dtype=float).reshape(0, nmeas)
        TU = np.eye(ncon, order='F', dtype=float)
        TY = np.eye(nmeas, order='F', dtype=float)

        AK, BK, CK, DK, info = sb10wd(
            n, m, np_, ncon, nmeas,
            A, B, C, D, F, H, TU, TY
        )

        assert info == 0
        assert AK.shape == (0, 0)
        assert BK.shape == (0, nmeas)
        assert CK.shape == (ncon, 0)
        assert DK.shape == (ncon, nmeas)

    def test_identity_transformations(self):
        """
        Test with identity transformation matrices TU=I and TY=I.

        Random seed: 456 (for reproducibility)
        """
        np.random.seed(456)

        n = 3
        m = 3
        np_ = 3
        ncon = 1
        nmeas = 1

        A = np.random.randn(n, n).astype(float, order='F')
        B = np.random.randn(n, m).astype(float, order='F')
        C = np.random.randn(np_, n).astype(float, order='F')
        D = np.random.randn(np_, m).astype(float, order='F')
        F = np.random.randn(ncon, n).astype(float, order='F')
        H = np.random.randn(n, nmeas).astype(float, order='F')
        TU = np.eye(ncon, order='F', dtype=float)
        TY = np.eye(nmeas, order='F', dtype=float)

        m1 = m - ncon
        np1 = np_ - nmeas

        B2 = B[:, m1:m]
        C2 = C[np1:np_, :]
        D22 = D[np1:np_, m1:m]

        AK_expected = A + H @ C2 + B2 @ F + H @ D22 @ F
        BK_expected = -H
        CK_expected = F
        DK_expected = np.zeros((ncon, nmeas), order='F', dtype=float)

        AK, BK, CK, DK, info = sb10wd(
            n, m, np_, ncon, nmeas,
            A, B, C, D, F, H, TU, TY
        )

        assert info == 0
        np.testing.assert_allclose(AK, AK_expected, rtol=1e-14)
        np.testing.assert_allclose(BK, BK_expected, rtol=1e-14)
        np.testing.assert_allclose(CK, CK_expected, rtol=1e-14)
        np.testing.assert_allclose(DK, DK_expected, rtol=1e-14)


class TestSB10WDMathematical:
    """Mathematical property validation tests."""

    def test_dk_always_zero(self):
        """
        Verify DK is always zero matrix for any inputs.

        Random seed: 789 (for reproducibility)
        """
        np.random.seed(789)

        n = 5
        m = 4
        np_ = 4
        ncon = 2
        nmeas = 2

        A = np.random.randn(n, n).astype(float, order='F')
        B = np.random.randn(n, m).astype(float, order='F')
        C = np.random.randn(np_, n).astype(float, order='F')
        D = np.random.randn(np_, m).astype(float, order='F')
        F = np.random.randn(ncon, n).astype(float, order='F')
        H = np.random.randn(n, nmeas).astype(float, order='F')
        TU = np.random.randn(ncon, ncon).astype(float, order='F')
        TY = np.random.randn(nmeas, nmeas).astype(float, order='F')

        AK, BK, CK, DK, info = sb10wd(
            n, m, np_, ncon, nmeas,
            A, B, C, D, F, H, TU, TY
        )

        assert info == 0
        np.testing.assert_allclose(
            DK, np.zeros((ncon, nmeas)), rtol=1e-14, atol=1e-15
        )

    def test_ck_equals_tu_times_f(self):
        """
        Verify CK = TU * F mathematical property.

        Random seed: 111 (for reproducibility)
        """
        np.random.seed(111)

        n = 4
        m = 3
        np_ = 3
        ncon = 2
        nmeas = 1

        A = np.random.randn(n, n).astype(float, order='F')
        B = np.random.randn(n, m).astype(float, order='F')
        C = np.random.randn(np_, n).astype(float, order='F')
        D = np.random.randn(np_, m).astype(float, order='F')
        F = np.random.randn(ncon, n).astype(float, order='F')
        H = np.random.randn(n, nmeas).astype(float, order='F')
        TU = np.random.randn(ncon, ncon).astype(float, order='F')
        TY = np.random.randn(nmeas, nmeas).astype(float, order='F')

        AK, BK, CK, DK, info = sb10wd(
            n, m, np_, ncon, nmeas,
            A, B, C, D, F, H, TU, TY
        )

        assert info == 0
        CK_expected = TU @ F
        np.testing.assert_allclose(CK, CK_expected, rtol=1e-14)

    def test_bk_equals_neg_h_times_ty(self):
        """
        Verify BK = -H * TY mathematical property.

        Random seed: 222 (for reproducibility)
        """
        np.random.seed(222)

        n = 3
        m = 4
        np_ = 4
        ncon = 2
        nmeas = 2

        A = np.random.randn(n, n).astype(float, order='F')
        B = np.random.randn(n, m).astype(float, order='F')
        C = np.random.randn(np_, n).astype(float, order='F')
        D = np.random.randn(np_, m).astype(float, order='F')
        F = np.random.randn(ncon, n).astype(float, order='F')
        H = np.random.randn(n, nmeas).astype(float, order='F')
        TU = np.random.randn(ncon, ncon).astype(float, order='F')
        TY = np.random.randn(nmeas, nmeas).astype(float, order='F')

        AK, BK, CK, DK, info = sb10wd(
            n, m, np_, ncon, nmeas,
            A, B, C, D, F, H, TU, TY
        )

        assert info == 0
        BK_expected = -H @ TY
        np.testing.assert_allclose(BK, BK_expected, rtol=1e-14)


class TestSB10WDMIMO:
    """MIMO tests for SB10WD."""

    def test_mimo_ncon2_nmeas2(self):
        """Test MIMO system with ncon=2, nmeas=2, formula verification."""
        np.random.seed(444)

        n = 4
        m = 5
        np_ = 5
        ncon = 2
        nmeas = 2
        m1 = m - ncon
        np1 = np_ - nmeas

        A = np.array([[-1, 0.1, 0, 0],
                       [0, -2, 0.1, 0],
                       [0, 0, -3, 0.1],
                       [0, 0, 0, -4]], dtype=float, order='F')

        B = np.random.randn(n, m).astype(float, order='F')
        C = np.random.randn(np_, n).astype(float, order='F')
        D = np.random.randn(np_, m).astype(float, order='F')
        F = np.random.randn(ncon, n).astype(float, order='F')
        H = np.random.randn(n, nmeas).astype(float, order='F')
        TU = np.random.randn(ncon, ncon).astype(float, order='F')
        TY = np.random.randn(nmeas, nmeas).astype(float, order='F')

        B2 = B[:, m1:m]
        C2 = C[np1:np_, :]
        D22 = D[np1:np_, m1:m]

        AK_expected = A + H @ C2 + B2 @ F + H @ D22 @ F
        BK_expected = -H @ TY
        CK_expected = TU @ F
        DK_expected = np.zeros((ncon, nmeas), order='F', dtype=float)

        AK, BK, CK, DK, info = sb10wd(
            n, m, np_, ncon, nmeas,
            A, B, C, D, F, H, TU, TY
        )

        assert info == 0
        np.testing.assert_allclose(AK, AK_expected, rtol=1e-14)
        np.testing.assert_allclose(BK, BK_expected, rtol=1e-14)
        np.testing.assert_allclose(CK, CK_expected, rtol=1e-14)
        np.testing.assert_allclose(DK, DK_expected, rtol=1e-14, atol=1e-15)


class TestSB10WDErrors:
    """Error handling tests for SB10WD."""

    def test_invalid_n(self):
        """Test error when n < 0."""
        with pytest.raises(ValueError, match="n must be >= 0"):
            sb10wd(
                -1, 2, 2, 1, 1,
                np.zeros((1, 1), order='F'),
                np.zeros((1, 2), order='F'),
                np.zeros((2, 1), order='F'),
                np.zeros((2, 2), order='F'),
                np.zeros((1, 1), order='F'),
                np.zeros((1, 1), order='F'),
                np.zeros((1, 1), order='F'),
                np.zeros((1, 1), order='F')
            )

    def test_invalid_ncon_greater_than_m(self):
        """Test error when ncon > m."""
        with pytest.raises(ValueError, match="ncon"):
            sb10wd(
                2, 2, 4, 3, 2,  # ncon=3 > m=2
                np.zeros((2, 2), order='F'),
                np.zeros((2, 2), order='F'),
                np.zeros((4, 2), order='F'),
                np.zeros((4, 2), order='F'),
                np.zeros((3, 2), order='F'),
                np.zeros((2, 2), order='F'),
                np.zeros((3, 3), order='F'),
                np.zeros((2, 2), order='F')
            )

    def test_invalid_nmeas_greater_than_np(self):
        """Test error when nmeas > np."""
        with pytest.raises(ValueError, match="nmeas"):
            sb10wd(
                2, 4, 2, 2, 3,  # nmeas=3 > np=2
                np.zeros((2, 2), order='F'),
                np.zeros((2, 4), order='F'),
                np.zeros((2, 2), order='F'),
                np.zeros((2, 4), order='F'),
                np.zeros((2, 2), order='F'),
                np.zeros((2, 3), order='F'),
                np.zeros((2, 2), order='F'),
                np.zeros((3, 3), order='F')
            )
