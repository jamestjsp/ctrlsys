"""
Tests for SB16AD: Frequency-weighted controller reduction.

SB16AD computes a reduced order controller (Acr,Bcr,Ccr,Dcr) for an
original state-space controller representation (Ac,Bc,Cc,Dc) using
frequency-weighted square-root or balancing-free square-root
Balance & Truncate (B&T) or Singular Perturbation Approximation (SPA)
model reduction methods.

Test data from SLICOT HTML documentation example.
"""

import numpy as np
import pytest

from slicot import sb16ad


class TestSB16ADBasic:
    """Basic functionality tests from HTML doc example."""

    def test_html_example_continuous_input_weighting(self):
        """
        Validate continuous-time controller reduction with input weighting.

        Test data from SLICOT HTML documentation example:
        - Continuous-time (DICO='C')
        - Standard Enns' method for controllability (JOBC='S')
        - Standard Enns' method for observability (JOBO='S')
        - Balancing-free B&T method (JOBMR='F')
        - Right (input) stability weighting (WEIGHT='I')
        - No equilibration (EQUIL='N')
        - Fixed order (ORDSEL='F')
        """
        # Open-loop system A, B, C, D (3x3, 3x1, 1x3, 1x1)
        n, m, p = 3, 1, 1
        a = np.array([
            [-1.0, 0.0, 4.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, -3.0]
        ], dtype=float, order='F')

        b = np.array([
            [1.0],
            [1.0],
            [1.0]
        ], dtype=float, order='F')

        c = np.array([[1.0, 1.0, 1.0]], dtype=float, order='F')

        d = np.array([[0.0]], dtype=float, order='F')

        # Controller K (3rd order): (Ac, Bc, Cc, Dc)
        nc = 3
        ac = np.array([
            [-26.4000, 6.4023, 4.3868],
            [32.0000, 0.0, 0.0],
            [0.0, 8.0000, 0.0]
        ], dtype=float, order='F')

        bc = np.array([
            [-16.0],
            [0.0],
            [0.0]
        ], dtype=float, order='F')

        cc = np.array([[9.2994, 1.1624, 0.1090]], dtype=float, order='F')

        dc = np.array([[0.0]], dtype=float, order='F')

        # Desired reduced order
        ncr = 2
        alpha = 0.0
        tol1 = 0.1
        tol2 = 0.0

        # Call sb16ad
        (acr, bcr, ccr, dcr, ncr_out, ncs, hsvc, iwarn, info) = sb16ad(
            dico='C',
            jobc='S',
            jobo='S',
            jobmr='F',
            weight='I',
            equil='N',
            ordsel='F',
            n=n,
            m=m,
            p=p,
            nc=nc,
            ncr=ncr,
            alpha=alpha,
            a=a,
            b=b,
            c=c,
            d=d,
            ac=ac,
            bc=bc,
            cc=cc,
            dc=dc,
            tol1=tol1,
            tol2=tol2
        )

        assert info == 0, f"sb16ad returned info={info}"
        assert iwarn == 0, f"sb16ad returned iwarn={iwarn}"
        assert ncr_out == 2, f"Expected ncr=2, got {ncr_out}"

        # Expected Hankel singular values from HTML doc
        hsvc_expected = np.array([3.8253, 0.2005], dtype=float)
        np.testing.assert_allclose(hsvc[:ncs], hsvc_expected, rtol=1e-3, atol=1e-4)

        # Expected reduced controller matrices from HTML doc
        acr_expected = np.array([
            [9.1900, 0.0000],
            [0.0000, -34.5297]
        ], dtype=float, order='F')

        bcr_expected = np.array([
            [-11.9593],
            [86.3137]
        ], dtype=float, order='F')

        ccr_expected = np.array([[2.8955, -1.3566]], dtype=float, order='F')

        dcr_expected = np.array([[0.0]], dtype=float, order='F')

        # Validate reduced controller (use looser tolerance due to HTML precision)
        # Note: State-space realizations are unique only up to similarity transformations.
        # For diagonal Acr, the states can have sign ambiguity (T = diag(±1, ±1)).
        # Check Acr exactly, but use absolute values for Bcr/Ccr since transfer function
        # H(s) = Ccr*(sI-Acr)^(-1)*Bcr is preserved when signs are flipped consistently.
        np.testing.assert_allclose(acr[:ncr_out, :ncr_out], acr_expected, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(np.abs(bcr[:ncr_out, :p]), np.abs(bcr_expected), rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(np.abs(ccr[:m, :ncr_out]), np.abs(ccr_expected), rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(dcr[:m, :p], dcr_expected, rtol=1e-3, atol=1e-4)


class TestSB16ADOrderSelection:
    """Test automatic order selection."""

    def test_automatic_order_selection(self):
        """
        Test automatic order selection based on tolerance.

        Uses same system as basic test but with ORDSEL='A'.
        """
        n, m, p = 3, 1, 1
        a = np.array([
            [-1.0, 0.0, 4.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, -3.0]
        ], dtype=float, order='F')

        b = np.array([
            [1.0],
            [1.0],
            [1.0]
        ], dtype=float, order='F')

        c = np.array([[1.0, 1.0, 1.0]], dtype=float, order='F')

        d = np.array([[0.0]], dtype=float, order='F')

        nc = 3
        ac = np.array([
            [-26.4000, 6.4023, 4.3868],
            [32.0000, 0.0, 0.0],
            [0.0, 8.0000, 0.0]
        ], dtype=float, order='F')

        bc = np.array([
            [-16.0],
            [0.0],
            [0.0]
        ], dtype=float, order='F')

        cc = np.array([[9.2994, 1.1624, 0.1090]], dtype=float, order='F')

        dc = np.array([[0.0]], dtype=float, order='F')

        ncr = 0  # Will be determined automatically
        alpha = 0.0
        tol1 = 1.0  # High tolerance - should give small order
        tol2 = 0.0

        (acr, bcr, ccr, dcr, ncr_out, ncs, hsvc, iwarn, info) = sb16ad(
            dico='C',
            jobc='S',
            jobo='S',
            jobmr='F',
            weight='I',
            equil='N',
            ordsel='A',
            n=n,
            m=m,
            p=p,
            nc=nc,
            ncr=ncr,
            alpha=alpha,
            a=a,
            b=b,
            c=c,
            d=d,
            ac=ac,
            bc=bc,
            cc=cc,
            dc=dc,
            tol1=tol1,
            tol2=tol2
        )

        assert info == 0, f"sb16ad returned info={info}"
        # With tol1=1.0, should keep only HSVs > 1.0, which is likely 1 or 2
        assert 0 <= ncr_out <= nc


class TestSB16ADNoWeighting:
    """Test without frequency weighting."""

    def test_no_weighting(self):
        """
        Test reduction without frequency weighting (WEIGHT='N').
        """
        n, m, p = 3, 1, 1
        a = np.array([
            [-1.0, 0.0, 4.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, -3.0]
        ], dtype=float, order='F')

        b = np.array([
            [1.0],
            [1.0],
            [1.0]
        ], dtype=float, order='F')

        c = np.array([[1.0, 1.0, 1.0]], dtype=float, order='F')

        d = np.array([[0.0]], dtype=float, order='F')

        nc = 3
        ac = np.array([
            [-26.4000, 6.4023, 4.3868],
            [32.0000, 0.0, 0.0],
            [0.0, 8.0000, 0.0]
        ], dtype=float, order='F')

        bc = np.array([
            [-16.0],
            [0.0],
            [0.0]
        ], dtype=float, order='F')

        cc = np.array([[9.2994, 1.1624, 0.1090]], dtype=float, order='F')

        dc = np.array([[0.0]], dtype=float, order='F')

        ncr = 2
        alpha = 0.0
        tol1 = 0.0
        tol2 = 0.0

        (acr, bcr, ccr, dcr, ncr_out, ncs, hsvc, iwarn, info) = sb16ad(
            dico='C',
            jobc='S',
            jobo='S',
            jobmr='B',  # Square-root B&T method
            weight='N',  # No weighting
            equil='N',
            ordsel='F',
            n=n,
            m=m,
            p=p,
            nc=nc,
            ncr=ncr,
            alpha=alpha,
            a=a,
            b=b,
            c=c,
            d=d,
            ac=ac,
            bc=bc,
            cc=cc,
            dc=dc,
            tol1=tol1,
            tol2=tol2
        )

        assert info == 0, f"sb16ad returned info={info}"
        assert ncr_out >= 0


class TestSB16ADEdgeCases:
    """Edge case tests."""

    def test_zero_order_controller(self):
        """Test with nc=0 (static controller)."""
        n, m, p = 2, 1, 1
        a = np.array([[-1.0, 0.0], [0.0, -2.0]], dtype=float, order='F')
        b = np.array([[1.0], [1.0]], dtype=float, order='F')
        c = np.array([[1.0, 1.0]], dtype=float, order='F')
        d = np.array([[0.0]], dtype=float, order='F')

        nc = 0
        ac = np.zeros((1, 1), dtype=float, order='F')  # Dummy
        bc = np.zeros((1, 1), dtype=float, order='F')
        cc = np.zeros((1, 1), dtype=float, order='F')
        dc = np.array([[1.0]], dtype=float, order='F')  # Static gain

        (acr, bcr, ccr, dcr, ncr_out, ncs, hsvc, iwarn, info) = sb16ad(
            dico='C',
            jobc='S',
            jobo='S',
            jobmr='B',
            weight='N',
            equil='N',
            ordsel='F',
            n=n,
            m=m,
            p=p,
            nc=nc,
            ncr=0,
            alpha=0.0,
            a=a,
            b=b,
            c=c,
            d=d,
            ac=ac,
            bc=bc,
            cc=cc,
            dc=dc,
            tol1=0.0,
            tol2=0.0
        )

        assert info == 0
        assert ncr_out == 0
        assert ncs == 0


class TestSB16ADMIMO:
    """MIMO controller reduction tests."""

    def test_mimo_controller_reduction(self):
        """
        MIMO system: 4-state open-loop, 2 inputs, 2 outputs.
        Reduce a 4th-order MIMO controller to 2nd order.
        """
        n, m, p = 4, 2, 2

        a = np.array([
            [-1.0,  0.5,  0.0,  0.0],
            [ 0.0, -2.0,  0.3,  0.0],
            [ 0.0,  0.0, -3.0,  0.1],
            [ 0.0,  0.0,  0.0, -4.0]
        ], dtype=float, order='F')

        b = np.array([
            [1.0, 0.0],
            [0.5, 1.0],
            [0.0, 0.5],
            [0.2, 0.3]
        ], dtype=float, order='F')

        c = np.array([
            [1.0, 0.5, 0.0, 0.0],
            [0.0, 1.0, 0.5, 0.2]
        ], dtype=float, order='F')

        d = np.zeros((p, m), dtype=float, order='F')

        nc = 4
        ac = np.array([
            [-5.0,  1.0,  0.0,  0.0],
            [-2.0, -6.0,  1.0,  0.0],
            [ 0.0, -1.0, -7.0,  0.5],
            [ 0.0,  0.0, -0.5, -8.0]
        ], dtype=float, order='F')

        bc = np.array([
            [2.0, 0.0],
            [0.0, 2.0],
            [1.0, 1.0],
            [0.5, 0.5]
        ], dtype=float, order='F')

        cc = np.array([
            [1.0, 0.5, 0.2, 0.1],
            [0.0, 1.0, 0.5, 0.2]
        ], dtype=float, order='F')

        dc = np.zeros((m, p), dtype=float, order='F')

        ncr = 2

        (acr, bcr, ccr, dcr, ncr_out, ncs, hsvc, iwarn, info) = sb16ad(
            dico='C',
            jobc='S',
            jobo='S',
            jobmr='B',
            weight='N',
            equil='N',
            ordsel='F',
            n=n,
            m=m,
            p=p,
            nc=nc,
            ncr=ncr,
            alpha=0.0,
            a=a,
            b=b,
            c=c,
            d=d,
            ac=ac,
            bc=bc,
            cc=cc,
            dc=dc,
            tol1=0.0,
            tol2=0.0
        )

        assert info == 0, f"sb16ad returned info={info}"
        assert ncr_out == 2
        assert ncs == nc

        for i in range(ncs - 1):
            assert hsvc[i] >= hsvc[i + 1] - 1e-14, "HSV not decreasing"


class TestSB16ADErrorHandling:
    """Test error handling."""

    def test_invalid_dico(self):
        """Test with invalid DICO parameter."""
        n, m, p = 2, 1, 1
        nc = 2
        a = np.eye(n, dtype=float, order='F')
        b = np.ones((n, m), dtype=float, order='F')
        c = np.ones((p, n), dtype=float, order='F')
        d = np.zeros((p, m), dtype=float, order='F')
        ac = -np.eye(nc, dtype=float, order='F')
        bc = np.ones((nc, p), dtype=float, order='F')
        cc = np.ones((m, nc), dtype=float, order='F')
        dc = np.zeros((m, p), dtype=float, order='F')

        with pytest.raises(ValueError):
            sb16ad(
                dico='X',  # Invalid
                jobc='S',
                jobo='S',
                jobmr='B',
                weight='N',
                equil='N',
                ordsel='F',
                n=n,
                m=m,
                p=p,
                nc=nc,
                ncr=1,
                alpha=0.0,
                a=a,
                b=b,
                c=c,
                d=d,
                ac=ac,
                bc=bc,
                cc=cc,
                dc=dc,
                tol1=0.0,
                tol2=0.0
            )

    def test_negative_dimensions(self):
        """Test with negative dimensions."""
        a = np.eye(2, dtype=float, order='F')
        b = np.ones((2, 1), dtype=float, order='F')
        c = np.ones((1, 2), dtype=float, order='F')
        d = np.zeros((1, 1), dtype=float, order='F')
        ac = -np.eye(2, dtype=float, order='F')
        bc = np.ones((2, 1), dtype=float, order='F')
        cc = np.ones((1, 2), dtype=float, order='F')
        dc = np.zeros((1, 1), dtype=float, order='F')

        with pytest.raises(ValueError):
            sb16ad(
                dico='C',
                jobc='S',
                jobo='S',
                jobmr='B',
                weight='N',
                equil='N',
                ordsel='F',
                n=-1,  # Invalid
                m=1,
                p=1,
                nc=2,
                ncr=1,
                alpha=0.0,
                a=a,
                b=b,
                c=c,
                d=d,
                ac=ac,
                bc=bc,
                cc=cc,
                dc=dc,
                tol1=0.0,
                tol2=0.0
            )
