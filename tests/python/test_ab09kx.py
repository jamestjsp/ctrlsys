"""
Tests for AB09KX - Stable projection of V*G*W or conj(V)*G*conj(W).

AB09KX constructs a state-space representation (A,BS,CS,DS) of the stable
projection of V*G*W or conj(V)*G*conj(W) from the state-space representations
(A,B,C,D), (AV,BV,CV,DV), and (AW,BW,CW,DW) of transfer-function matrices G, V
and W respectively.

Requirements:
- G must be stable, A must be in real Schur form
- For JOB='N' (V*G*W): V and W must be completely unstable
- For JOB='C' (conj(V)*G*conj(W)): V and W must be stable

Mode Parameters:
- JOB: 'N' (compute V*G*W), 'C' (compute conj(V)*G*conj(W))
- DICO: 'C' (continuous-time), 'D' (discrete-time)
- WEIGHT: 'N' (no weights), 'L' (left only), 'R' (right only), 'B' (both)

Warning codes (IWARN):
- 0: no warning
- 1: AV not completely unstable (JOB='N') or not stable (JOB='C')
- 2: AW not completely unstable (JOB='N') or not stable (JOB='C')
- 3: both warnings apply

Error codes (INFO):
- 0: success
- 1: reduction of AV to Schur form failed
- 2: reduction of AW to Schur form failed
- 3: Sylvester equation failed (A and AV have common eigenvalues)
- 4: Sylvester equation failed (A and AW have common eigenvalues)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


class TestAB09KXLeftWeighting:
    """Tests with left weighting only (WEIGHT='L')."""

    def test_continuous_left_weight_job_n(self):
        """
        Test continuous-time left weighting with JOB='N' (compute V*G).

        Random seed: 42 (for reproducibility)
        System G stable (diagonal A), V unstable (positive eigenvalues).
        """
        from ctrlsys import ab09kx

        np.random.seed(42)
        n, m, p = 2, 1, 2
        nv = 2

        a = np.array([
            [-1.0, 0.0],
            [0.0, -2.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0],
            [0.5]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        d = np.array([
            [0.1],
            [0.2]
        ], order='F', dtype=float)

        av = np.array([
            [1.0, 0.0],
            [0.0, 2.0]
        ], order='F', dtype=float)

        bv = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        cv = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        dv = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, m), order='F', dtype=float)
        cw = np.zeros((m, 1), order='F', dtype=float)
        dw = np.zeros((m, m), order='F', dtype=float)

        b_out, c_out, d_out, av_out, bv_out, cv_out, aw_out, bw_out, cw_out, iwarn, info = ab09kx(
            'N', 'C', 'L', n, nv, 0, m, p,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw
        )

        assert info == 0, f"Expected info=0, got {info}"
        assert iwarn == 0, f"Expected iwarn=0, got {iwarn}"

        assert b_out.shape == (n, m)
        assert c_out.shape == (p, n)
        assert d_out.shape == (p, m)

    def test_continuous_left_weight_job_c(self):
        """
        Test continuous-time left weighting with JOB='C' (compute conj(V)*G).

        Random seed: 123 (for reproducibility)
        Both G and V must be stable for JOB='C'.
        """
        from ctrlsys import ab09kx

        np.random.seed(123)
        n, m, p = 2, 1, 2
        nv = 2

        a = np.array([
            [-1.0, 0.0],
            [0.0, -2.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0],
            [0.5]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        d = np.array([
            [0.1],
            [0.2]
        ], order='F', dtype=float)

        av = np.array([
            [-1.0, 0.0],
            [0.0, -2.0]
        ], order='F', dtype=float)

        bv = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        cv = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        dv = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, m), order='F', dtype=float)
        cw = np.zeros((m, 1), order='F', dtype=float)
        dw = np.zeros((m, m), order='F', dtype=float)

        b_out, c_out, d_out, av_out, bv_out, cv_out, aw_out, bw_out, cw_out, iwarn, info = ab09kx(
            'C', 'C', 'L', n, nv, 0, m, p,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw
        )

        assert info == 0, f"Expected info=0, got {info}"
        assert iwarn == 0, f"Expected iwarn=0, got {iwarn}"


class TestAB09KXRightWeighting:
    """Tests with right weighting only (WEIGHT='R')."""

    def test_continuous_right_weight_job_n(self):
        """
        Test continuous-time right weighting with JOB='N' (compute G*W).

        Random seed: 456 (for reproducibility)
        System G stable (diagonal A), W unstable (positive eigenvalues).
        """
        from ctrlsys import ab09kx

        np.random.seed(456)
        n, m, p = 2, 2, 1
        nw = 2

        a = np.array([
            [-1.0, 0.0],
            [0.0, -2.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.5]
        ], order='F', dtype=float)

        d = np.array([
            [0.1, 0.2]
        ], order='F', dtype=float)

        av = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, p), order='F', dtype=float)
        cv = np.zeros((p, 1), order='F', dtype=float)
        dv = np.zeros((p, p), order='F', dtype=float)

        aw = np.array([
            [1.0, 0.0],
            [0.0, 2.0]
        ], order='F', dtype=float)

        bw = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        cw = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        dw = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        b_out, c_out, d_out, av_out, bv_out, cv_out, aw_out, bw_out, cw_out, iwarn, info = ab09kx(
            'N', 'C', 'R', n, 0, nw, m, p,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw
        )

        assert info == 0, f"Expected info=0, got {info}"
        assert iwarn == 0, f"Expected iwarn=0, got {iwarn}"


class TestAB09KXBothWeighting:
    """Tests with both left and right weighting (WEIGHT='B')."""

    def test_continuous_both_weights_job_n(self):
        """
        Test continuous-time both weights with JOB='N' (compute V*G*W).

        Random seed: 789 (for reproducibility)
        G stable, V and W unstable.
        """
        from ctrlsys import ab09kx

        np.random.seed(789)
        n, m, p = 2, 2, 2
        nv, nw = 1, 1

        a = np.array([
            [-1.0, 0.0],
            [0.0, -2.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        d = np.array([
            [0.1, 0.0],
            [0.0, 0.1]
        ], order='F', dtype=float)

        av = np.array([[1.0]], order='F', dtype=float)
        bv = np.array([[1.0, 0.0]], order='F', dtype=float)
        cv = np.array([[1.0], [0.0]], order='F', dtype=float)
        dv = np.eye(p, order='F', dtype=float)

        aw = np.array([[2.0]], order='F', dtype=float)
        bw = np.array([[1.0, 0.0]], order='F', dtype=float)
        cw = np.array([[1.0], [0.0]], order='F', dtype=float)
        dw = np.eye(m, order='F', dtype=float)

        b_out, c_out, d_out, av_out, bv_out, cv_out, aw_out, bw_out, cw_out, iwarn, info = ab09kx(
            'N', 'C', 'B', n, nv, nw, m, p,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw
        )

        assert info == 0, f"Expected info=0, got {info}"
        assert iwarn == 0, f"Expected iwarn=0, got {iwarn}"


class TestAB09KXNoWeighting:
    """Tests with no weighting (WEIGHT='N')."""

    def test_no_weighting_quick_return(self):
        """
        Test no weighting (WEIGHT='N') - quick return case.

        When WEIGHT='N', V=I and W=I, so no computation needed.
        """
        from ctrlsys import ab09kx

        n, m, p = 2, 1, 1

        a = np.array([
            [-1.0, 0.0],
            [0.0, -2.0]
        ], order='F', dtype=float)

        b = np.array([[1.0], [0.5]], order='F', dtype=float)
        c = np.array([[1.0, 0.5]], order='F', dtype=float)
        d = np.array([[0.1]], order='F', dtype=float)

        av = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, p), order='F', dtype=float)
        cv = np.zeros((p, 1), order='F', dtype=float)
        dv = np.zeros((p, p), order='F', dtype=float)

        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, m), order='F', dtype=float)
        cw = np.zeros((m, 1), order='F', dtype=float)
        dw = np.zeros((m, m), order='F', dtype=float)

        b_out, c_out, d_out, av_out, bv_out, cv_out, aw_out, bw_out, cw_out, iwarn, info = ab09kx(
            'N', 'C', 'N', n, 0, 0, m, p,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw
        )

        assert info == 0, f"Expected info=0, got {info}"


class TestAB09KXDiscreteTime:
    """Tests for discrete-time systems (DICO='D')."""

    def test_discrete_left_weight_job_n(self):
        """
        Test discrete-time left weighting with JOB='N'.

        Random seed: 111 (for reproducibility)
        G stable (|eigenvalues| < 1), V unstable (|eigenvalues| > 1).
        """
        from ctrlsys import ab09kx

        np.random.seed(111)
        n, m, p = 2, 1, 2
        nv = 2

        a = np.array([
            [0.5, 0.0],
            [0.0, 0.3]
        ], order='F', dtype=float)

        b = np.array([
            [1.0],
            [0.5]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        d = np.array([
            [0.1],
            [0.2]
        ], order='F', dtype=float)

        av = np.array([
            [1.5, 0.0],
            [0.0, 2.0]
        ], order='F', dtype=float)

        bv = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        cv = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        dv = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, m), order='F', dtype=float)
        cw = np.zeros((m, 1), order='F', dtype=float)
        dw = np.zeros((m, m), order='F', dtype=float)

        b_out, c_out, d_out, av_out, bv_out, cv_out, aw_out, bw_out, cw_out, iwarn, info = ab09kx(
            'N', 'D', 'L', n, nv, 0, m, p,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw
        )

        assert info == 0, f"Expected info=0, got {info}"
        assert iwarn == 0, f"Expected iwarn=0, got {iwarn}"

    def test_discrete_left_weight_job_c(self):
        """
        Test discrete-time left weighting with JOB='C' (conj(V)*G).

        Random seed: 222 (for reproducibility)
        Both G and V stable (|eigenvalues| < 1) for JOB='C'.
        """
        from ctrlsys import ab09kx

        np.random.seed(222)
        n, m, p = 2, 1, 2
        nv = 2

        a = np.array([
            [0.5, 0.0],
            [0.0, 0.3]
        ], order='F', dtype=float)

        b = np.array([
            [1.0],
            [0.5]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        d = np.array([
            [0.1],
            [0.2]
        ], order='F', dtype=float)

        av = np.array([
            [0.4, 0.0],
            [0.0, 0.2]
        ], order='F', dtype=float)

        bv = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        cv = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        dv = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, m), order='F', dtype=float)
        cw = np.zeros((m, 1), order='F', dtype=float)
        dw = np.zeros((m, m), order='F', dtype=float)

        b_out, c_out, d_out, av_out, bv_out, cv_out, aw_out, bw_out, cw_out, iwarn, info = ab09kx(
            'C', 'D', 'L', n, nv, 0, m, p,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw
        )

        assert info == 0, f"Expected info=0, got {info}"
        assert iwarn == 0, f"Expected iwarn=0, got {iwarn}"


class TestAB09KXMathematicalProperties:
    """Tests for mathematical properties of the stable projection."""

    def test_no_weight_preserves_system(self):
        """
        Validate that WEIGHT='N' preserves the system unchanged.

        When no weighting is applied (V=I, W=I), the output should equal input.
        """
        from ctrlsys import ab09kx

        n, m, p = 2, 1, 2

        a = np.array([
            [-1.0, 0.0],
            [0.0, -2.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0],
            [0.5]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        d = np.array([
            [0.5],
            [0.3]
        ], order='F', dtype=float)

        b_orig = b.copy()
        c_orig = c.copy()
        d_orig = d.copy()

        av = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, p), order='F', dtype=float)
        cv = np.zeros((p, 1), order='F', dtype=float)
        dv = np.zeros((p, p), order='F', dtype=float)

        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, m), order='F', dtype=float)
        cw = np.zeros((m, 1), order='F', dtype=float)
        dw = np.zeros((m, m), order='F', dtype=float)

        b_out, c_out, d_out, av_out, bv_out, cv_out, aw_out, bw_out, cw_out, iwarn, info = ab09kx(
            'N', 'C', 'N', n, 0, 0, m, p,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw
        )

        assert info == 0
        assert_allclose(b_out, b_orig, rtol=1e-14,
                       err_msg="B should be unchanged when WEIGHT='N'")
        assert_allclose(c_out, c_orig, rtol=1e-14,
                       err_msg="C should be unchanged when WEIGHT='N'")
        assert_allclose(d_out, d_orig, rtol=1e-14,
                       err_msg="D should be unchanged when WEIGHT='N'")

    def test_identity_weight_preserves_d(self):
        """
        Validate D matrix transformation with identity DV.

        When DV=I and NV=0, D is transformed only by DV multiplication.
        """
        from ctrlsys import ab09kx

        n, m, p = 2, 1, 2

        a = np.array([
            [-1.0, 0.0],
            [0.0, -2.0]
        ], order='F', dtype=float)

        b = np.array([
            [1.0],
            [0.5]
        ], order='F', dtype=float)

        c = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], order='F', dtype=float)

        d = np.array([
            [0.5],
            [0.3]
        ], order='F', dtype=float)

        d_orig = d.copy()

        av = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, p), order='F', dtype=float)
        cv = np.zeros((p, 1), order='F', dtype=float)
        dv = np.eye(p, order='F', dtype=float)

        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, m), order='F', dtype=float)
        cw = np.zeros((m, 1), order='F', dtype=float)
        dw = np.zeros((m, m), order='F', dtype=float)

        b_out, c_out, d_out, av_out, bv_out, cv_out, aw_out, bw_out, cw_out, iwarn, info = ab09kx(
            'N', 'C', 'L', n, 0, 0, m, p,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw
        )

        assert info == 0
        assert_allclose(d_out, d_orig, rtol=1e-14,
                       err_msg="D should be unchanged when NV=0 and DV=I")


class TestAB09KXWarnings:
    """Tests for warning conditions (IWARN)."""

    def test_warning_av_not_unstable(self):
        """
        Test IWARN=1 when AV has eigenvalues in stable region (JOB='N').

        For JOB='N', V should be completely unstable.
        """
        from ctrlsys import ab09kx

        n, m, p = 2, 1, 2
        nv = 2

        a = np.array([
            [-1.0, 0.0],
            [0.0, -2.0]
        ], order='F', dtype=float)

        b = np.array([[1.0], [0.5]], order='F', dtype=float)
        c = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)
        d = np.array([[0.1], [0.2]], order='F', dtype=float)

        av = np.array([
            [-0.5, 0.0],
            [0.0, 2.0]
        ], order='F', dtype=float)

        bv = np.eye(nv, p, order='F', dtype=float)
        cv = np.eye(p, nv, order='F', dtype=float)
        dv = np.eye(p, order='F', dtype=float)

        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, m), order='F', dtype=float)
        cw = np.zeros((m, 1), order='F', dtype=float)
        dw = np.zeros((m, m), order='F', dtype=float)

        b_out, c_out, d_out, av_out, bv_out, cv_out, aw_out, bw_out, cw_out, iwarn, info = ab09kx(
            'N', 'C', 'L', n, nv, 0, m, p,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw
        )

        assert info == 0
        assert iwarn == 1, f"Expected iwarn=1, got {iwarn}"

    def test_warning_av_not_stable_job_c(self):
        """
        Test IWARN=1 when AV is not stable for JOB='C'.

        For JOB='C', V should be stable.
        """
        from ctrlsys import ab09kx

        n, m, p = 2, 1, 2
        nv = 2

        a = np.array([
            [-1.0, 0.0],
            [0.0, -2.0]
        ], order='F', dtype=float)

        b = np.array([[1.0], [0.5]], order='F', dtype=float)
        c = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)
        d = np.array([[0.1], [0.2]], order='F', dtype=float)

        av = np.array([
            [0.5, 0.0],
            [0.0, -2.0]
        ], order='F', dtype=float)

        bv = np.eye(nv, p, order='F', dtype=float)
        cv = np.eye(p, nv, order='F', dtype=float)
        dv = np.eye(p, order='F', dtype=float)

        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, m), order='F', dtype=float)
        cw = np.zeros((m, 1), order='F', dtype=float)
        dw = np.zeros((m, m), order='F', dtype=float)

        b_out, c_out, d_out, av_out, bv_out, cv_out, aw_out, bw_out, cw_out, iwarn, info = ab09kx(
            'C', 'C', 'L', n, nv, 0, m, p,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw
        )

        assert info == 0
        assert iwarn == 1, f"Expected iwarn=1, got {iwarn}"


class TestAB09KXEdgeCases:
    """Edge case tests."""

    def test_n_zero(self):
        """Test with N=0 (zero-order G system)."""
        from ctrlsys import ab09kx

        n, m, p = 0, 1, 1

        a = np.zeros((1, 1), order='F', dtype=float)
        b = np.zeros((1, m), order='F', dtype=float)
        c = np.zeros((p, 1), order='F', dtype=float)
        d = np.array([[1.0]], order='F', dtype=float)

        av = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, p), order='F', dtype=float)
        cv = np.zeros((p, 1), order='F', dtype=float)
        dv = np.zeros((p, p), order='F', dtype=float)

        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, m), order='F', dtype=float)
        cw = np.zeros((m, 1), order='F', dtype=float)
        dw = np.zeros((m, m), order='F', dtype=float)

        b_out, c_out, d_out, av_out, bv_out, cv_out, aw_out, bw_out, cw_out, iwarn, info = ab09kx(
            'N', 'C', 'N', n, 0, 0, m, p,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw
        )

        assert info == 0

    def test_m_or_p_zero(self):
        """Test with M=0 or P=0 (quick return)."""
        from ctrlsys import ab09kx

        n, m, p = 2, 0, 1

        a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
        b = np.zeros((n, 1), order='F', dtype=float)
        c = np.zeros((1, n), order='F', dtype=float)
        d = np.zeros((1, 1), order='F', dtype=float)

        av = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, 1), order='F', dtype=float)
        cv = np.zeros((1, 1), order='F', dtype=float)
        dv = np.zeros((1, 1), order='F', dtype=float)

        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, 1), order='F', dtype=float)
        cw = np.zeros((1, 1), order='F', dtype=float)
        dw = np.zeros((1, 1), order='F', dtype=float)

        b_out, c_out, d_out, av_out, bv_out, cv_out, aw_out, bw_out, cw_out, iwarn, info = ab09kx(
            'N', 'C', 'N', n, 0, 0, m, p,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw
        )

        assert info == 0

    def test_nv_zero_with_left_weight(self):
        """
        Test with NV=0 and WEIGHT='L'.

        When NV=0 and left weighting requested, V acts as identity.
        """
        from ctrlsys import ab09kx

        n, m, p = 2, 1, 2

        a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
        b = np.array([[1.0], [0.5]], order='F', dtype=float)
        c = np.array([[1.0, 0.0], [0.0, 1.0]], order='F', dtype=float)
        d = np.array([[0.1], [0.2]], order='F', dtype=float)

        av = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, p), order='F', dtype=float)
        cv = np.zeros((p, 1), order='F', dtype=float)
        dv = np.eye(p, order='F', dtype=float)

        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, m), order='F', dtype=float)
        cw = np.zeros((m, 1), order='F', dtype=float)
        dw = np.zeros((m, m), order='F', dtype=float)

        b_out, c_out, d_out, av_out, bv_out, cv_out, aw_out, bw_out, cw_out, iwarn, info = ab09kx(
            'N', 'C', 'L', n, 0, 0, m, p,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw
        )

        assert info == 0


class TestAB09KXErrors:
    """Error handling tests."""

    def test_invalid_job(self):
        """Test error for invalid JOB parameter."""
        from ctrlsys import ab09kx

        n, m, p = 2, 1, 1

        a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
        b = np.array([[1.0], [0.5]], order='F', dtype=float)
        c = np.array([[1.0, 0.5]], order='F', dtype=float)
        d = np.array([[0.1]], order='F', dtype=float)

        av = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, p), order='F', dtype=float)
        cv = np.zeros((p, 1), order='F', dtype=float)
        dv = np.zeros((p, p), order='F', dtype=float)

        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, m), order='F', dtype=float)
        cw = np.zeros((m, 1), order='F', dtype=float)
        dw = np.zeros((m, m), order='F', dtype=float)

        with pytest.raises(ValueError):
            ab09kx(
                'X', 'C', 'N', n, 0, 0, m, p,
                a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw
            )

    def test_invalid_dico(self):
        """Test error for invalid DICO parameter."""
        from ctrlsys import ab09kx

        n, m, p = 2, 1, 1

        a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
        b = np.array([[1.0], [0.5]], order='F', dtype=float)
        c = np.array([[1.0, 0.5]], order='F', dtype=float)
        d = np.array([[0.1]], order='F', dtype=float)

        av = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, p), order='F', dtype=float)
        cv = np.zeros((p, 1), order='F', dtype=float)
        dv = np.zeros((p, p), order='F', dtype=float)

        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, m), order='F', dtype=float)
        cw = np.zeros((m, 1), order='F', dtype=float)
        dw = np.zeros((m, m), order='F', dtype=float)

        with pytest.raises(ValueError):
            ab09kx(
                'N', 'X', 'N', n, 0, 0, m, p,
                a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw
            )

    def test_invalid_weight(self):
        """Test error for invalid WEIGHT parameter."""
        from ctrlsys import ab09kx

        n, m, p = 2, 1, 1

        a = np.array([[-1.0, 0.0], [0.0, -2.0]], order='F', dtype=float)
        b = np.array([[1.0], [0.5]], order='F', dtype=float)
        c = np.array([[1.0, 0.5]], order='F', dtype=float)
        d = np.array([[0.1]], order='F', dtype=float)

        av = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, p), order='F', dtype=float)
        cv = np.zeros((p, 1), order='F', dtype=float)
        dv = np.zeros((p, p), order='F', dtype=float)

        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, m), order='F', dtype=float)
        cw = np.zeros((m, 1), order='F', dtype=float)
        dw = np.zeros((m, m), order='F', dtype=float)

        with pytest.raises(ValueError):
            ab09kx(
                'N', 'C', 'X', n, 0, 0, m, p,
                a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw
            )

    def test_negative_n(self):
        """Test error for negative N."""
        from ctrlsys import ab09kx

        a = np.zeros((1, 1), order='F', dtype=float)
        b = np.zeros((1, 1), order='F', dtype=float)
        c = np.zeros((1, 1), order='F', dtype=float)
        d = np.zeros((1, 1), order='F', dtype=float)
        av = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, 1), order='F', dtype=float)
        cv = np.zeros((1, 1), order='F', dtype=float)
        dv = np.zeros((1, 1), order='F', dtype=float)
        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, 1), order='F', dtype=float)
        cw = np.zeros((1, 1), order='F', dtype=float)
        dw = np.zeros((1, 1), order='F', dtype=float)

        with pytest.raises((ValueError, RuntimeError)):
            ab09kx(
                'N', 'C', 'N', -1, 0, 0, 1, 1,
                a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw
            )
