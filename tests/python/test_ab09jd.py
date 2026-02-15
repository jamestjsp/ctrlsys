"""
Tests for AB09JD - Frequency-weighted Hankel-norm approximation with invertible weights.

AB09JD computes a reduced order model (Ar,Br,Cr,Dr) for an original
state-space representation (A,B,C,D) by using the frequency weighted
optimal Hankel-norm approximation method.

The Hankel norm of the weighted error op(V)*(G-Gr)*op(W) is minimized, where:
- G and Gr are the transfer-function matrices of original/reduced systems
- op(X) denotes X, inv(X), conj(X), or conj(inv(X))
- V and W are invertible transfer-function matrices for frequency weighting

Mode Parameters:
- JOBV: 'N' (V=I), 'V' (op(V)=V), 'I' (op(V)=inv(V)), 'C' (op(V)=conj(V)), 'R' (op(V)=conj(inv(V)))
- JOBW: 'N' (W=I), 'W' (op(W)=W), 'I' (op(W)=inv(W)), 'C' (op(W)=conj(W)), 'R' (op(W)=conj(inv(W)))
- JOBINV: 'N' (inverse-free descriptor), 'I' (inversion-based), 'A' (auto-switch)
- DICO: 'C' (continuous-time), 'D' (discrete-time)
- EQUIL: 'S' (equilibrate), 'N' (no equilibration)
- ORDSEL: 'F' (fixed order), 'A' (automatic order based on TOL1)

Warning codes (IWARN):
- 0: no warning
- 1: NR > NSMIN, NR set to NSMIN
- 2: NR < NU (order of unstable part), NR set to NU

Error codes (INFO):
- 0: success
- 1: ordered Schur form of A failed
- 2: separation of stable/unstable blocks failed
- 3: reduction of AV to Schur form failed
- 4: reduction of AW to Schur form failed
- 5: reduction to generalized Schur form for inv(V) failed
- 6: reduction to generalized Schur form for inv(W) failed
- 7: Hankel singular value computation failed
- 8: stable projection in Hankel-norm approximation failed
- 9: order mismatch in stable projection
- 10: reduction of AV-BV*inv(DV)*CV to Schur form failed
- 11: reduction of AW-BW*inv(DW)*CW to Schur form failed
- 12: Sylvester equation failed (V poles match G1 poles)
- 13: Sylvester equation failed (W poles match G1 poles)
- 14: Sylvester equation failed (V zeros match G1sr poles)
- 15: Sylvester equation failed (W zeros match G1sr poles)
- 16: generalized Sylvester failed (V zeros match G1sr poles)
- 17: generalized Sylvester failed (W zeros match G1sr poles)
- 18: op(V) is not antistable
- 19: op(W) is not antistable
- 20: V is not invertible
- 21: W is not invertible
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


class TestAB09JDDocExample:
    """Tests based on HTML documentation example."""

    def test_continuous_left_weight_automatic_order(self):
        """
        Test from SLICOT HTML documentation AB09JD.html.

        6th order continuous system with 2nd order left weighting.
        ORDSEL='A' for automatic order selection based on TOL1=0.1.
        JOBV='V' means op(V) = V (antistable V required).
        """
        from ctrlsys import ab09jd

        n, m, p = 6, 1, 1
        nv, nw = 2, 0

        a = np.array([
            [-3.8637, -7.4641, -9.1416, -7.4641, -3.8637, -1.0000],
            [ 1.0000,  0.0,     0.0,     0.0,     0.0,     0.0   ],
            [ 0.0,     1.0000,  0.0,     0.0,     0.0,     0.0   ],
            [ 0.0,     0.0,     1.0000,  0.0,     0.0,     0.0   ],
            [ 0.0,     0.0,     0.0,     1.0000,  0.0,     0.0   ],
            [ 0.0,     0.0,     0.0,     0.0,     1.0000,  0.0   ]
        ], order='F', dtype=float)

        b = np.array([
            [1.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0]
        ], order='F', dtype=float)

        c = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ], order='F', dtype=float)

        d = np.array([
            [0.0]
        ], order='F', dtype=float)

        av = np.array([
            [0.2, -1.0],
            [1.0,  0.0]
        ], order='F', dtype=float)

        bv = np.array([
            [1.0],
            [0.0]
        ], order='F', dtype=float)

        cv = np.array([
            [-1.8, 0.0]
        ], order='F', dtype=float)

        dv = np.array([
            [1.0]
        ], order='F', dtype=float)

        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, m), order='F', dtype=float)
        cw = np.zeros((m, 1), order='F', dtype=float)
        dw = np.zeros((m, m), order='F', dtype=float)

        nr = 0
        alpha = 0.0
        tol1 = 0.1
        tol2 = 1e-14

        (a_out, b_out, c_out, d_out, av_out, bv_out, cv_out,
         aw_out, bw_out, cw_out, nr_out, ns, hsv, iwarn, info) = ab09jd(
            'V', 'N', 'I', 'C', 'S', 'A', n, nv, nw, m, p, nr, alpha,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, tol1, tol2
        )

        assert info == 0, f"Expected info=0, got {info}"
        assert nr_out == 4, f"Expected nr=4, got {nr_out}"
        assert ns == 6, f"Expected ns=6, got {ns}"

        hsv_expected = np.array([2.6790, 2.1589, 0.8424, 0.1929, 0.0219, 0.0011])
        assert_allclose(hsv[:ns], hsv_expected, rtol=0.03, atol=1e-4,
                       err_msg="Hankel singular values mismatch")

        ar_expected = np.array([
            [-0.2391,  0.3072,  1.1630,  1.1967],
            [-2.9709, -0.2391,  2.6270,  3.1027],
            [ 0.0000,  0.0000, -0.5137, -1.2842],
            [ 0.0000,  0.0000,  0.1519, -0.5137]
        ], order='F', dtype=float)

        br_expected = np.array([
            [-1.0497],
            [-3.7052],
            [0.8223],
            [0.7435]
        ], order='F', dtype=float)

        cr_expected = np.array([
            [-0.4466, 0.0143, -0.4780, -0.2013]
        ], order='F', dtype=float)

        dr_expected = np.array([
            [0.0219]
        ], order='F', dtype=float)

        ar_actual = a_out[:nr_out, :nr_out].copy()
        br_actual = b_out[:nr_out, :m].copy()
        cr_actual = c_out[:p, :nr_out].copy()

        ar_actual = np.abs(ar_actual)
        ar_exp_abs = np.abs(ar_expected)
        assert_allclose(ar_actual, ar_exp_abs, rtol=5e-3, atol=1e-3,
                       err_msg="Reduced A matrix mismatch")

        br_actual = np.abs(br_actual)
        br_exp_abs = np.abs(br_expected)
        assert_allclose(br_actual, br_exp_abs, rtol=5e-3, atol=1e-3,
                       err_msg="Reduced B matrix mismatch")

        cr_actual = np.abs(cr_actual)
        cr_exp_abs = np.abs(cr_expected)
        assert_allclose(cr_actual, cr_exp_abs, rtol=5e-3, atol=1e-3,
                       err_msg="Reduced C matrix mismatch")

        assert_allclose(d_out[:p, :m], dr_expected, rtol=5e-3, atol=1e-3,
                       err_msg="Reduced D matrix mismatch")


class TestAB09JDNoWeighting:
    """Tests with no frequency weighting (JOBV='N', JOBW='N')."""

    def test_no_weight_stable_system(self):
        """
        Test with no frequency weighting (V=I, W=I).

        Random seed: 42 (for reproducibility)
        """
        from ctrlsys import ab09jd

        np.random.seed(42)
        n, m, p = 4, 1, 1
        nv, nw = 0, 0

        a = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        av = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, p), order='F', dtype=float)
        cv = np.zeros((p, 1), order='F', dtype=float)
        dv = np.zeros((p, p), order='F', dtype=float)

        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, m), order='F', dtype=float)
        cw = np.zeros((m, 1), order='F', dtype=float)
        dw = np.zeros((m, m), order='F', dtype=float)

        nr = 2
        alpha = 0.0
        tol1 = 0.0
        tol2 = 0.0

        (a_out, b_out, c_out, d_out, av_out, bv_out, cv_out,
         aw_out, bw_out, cw_out, nr_out, ns, hsv, iwarn, info) = ab09jd(
            'N', 'N', 'I', 'C', 'N', 'F', n, nv, nw, m, p, nr, alpha,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, tol1, tol2
        )

        assert info == 0, f"Expected info=0, got {info}"
        assert ns == n, f"Expected ns={n}, got {ns}"
        assert len(hsv) >= ns

        for i in range(ns - 1):
            assert hsv[i] >= hsv[i + 1], "HSV should be decreasingly ordered"


class TestAB09JDLeftWeighting:
    """Tests with left frequency weighting only."""

    def test_left_weight_jobv_v_continuous(self):
        """
        Test left weighting with JOBV='V' (op(V)=V).

        V must be antistable for JOBV='V'.
        Random seed: 123 (for reproducibility)
        """
        from ctrlsys import ab09jd

        np.random.seed(123)
        n, m, p = 3, 1, 1
        nv, nw = 1, 0

        a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')
        b = np.array([[1.0], [1.0], [1.0]], order='F', dtype=float)
        c = np.array([[1.0, 1.0, 1.0]], order='F', dtype=float)
        d = np.array([[0.0]], order='F', dtype=float)

        av = np.array([[1.0]], order='F', dtype=float)
        bv = np.array([[1.0]], order='F', dtype=float)
        cv = np.array([[1.0]], order='F', dtype=float)
        dv = np.array([[1.0]], order='F', dtype=float)

        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, m), order='F', dtype=float)
        cw = np.zeros((m, 1), order='F', dtype=float)
        dw = np.zeros((m, m), order='F', dtype=float)

        nr = 2
        alpha = 0.0
        tol1 = 0.0
        tol2 = 0.0

        (a_out, b_out, c_out, d_out, av_out, bv_out, cv_out,
         aw_out, bw_out, cw_out, nr_out, ns, hsv, iwarn, info) = ab09jd(
            'V', 'N', 'I', 'C', 'N', 'F', n, nv, nw, m, p, nr, alpha,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, tol1, tol2
        )

        assert info == 0, f"Expected info=0, got {info}"
        assert ns == n

    def test_left_weight_jobv_c_continuous(self):
        """
        Test left weighting with JOBV='C' (op(V)=conj(V)).

        V must be stable for JOBV='C'.
        Random seed: 456 (for reproducibility)
        """
        from ctrlsys import ab09jd

        np.random.seed(456)
        n, m, p = 3, 1, 1
        nv, nw = 1, 0

        a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')
        b = np.array([[1.0], [1.0], [1.0]], order='F', dtype=float)
        c = np.array([[1.0, 1.0, 1.0]], order='F', dtype=float)
        d = np.array([[0.0]], order='F', dtype=float)

        av = np.array([[-1.0]], order='F', dtype=float)
        bv = np.array([[1.0]], order='F', dtype=float)
        cv = np.array([[1.0]], order='F', dtype=float)
        dv = np.array([[1.0]], order='F', dtype=float)

        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, m), order='F', dtype=float)
        cw = np.zeros((m, 1), order='F', dtype=float)
        dw = np.zeros((m, m), order='F', dtype=float)

        nr = 2
        alpha = 0.0
        tol1 = 0.0
        tol2 = 0.0

        (a_out, b_out, c_out, d_out, av_out, bv_out, cv_out,
         aw_out, bw_out, cw_out, nr_out, ns, hsv, iwarn, info) = ab09jd(
            'C', 'N', 'I', 'C', 'N', 'F', n, nv, nw, m, p, nr, alpha,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, tol1, tol2
        )

        assert info == 0, f"Expected info=0, got {info}"
        assert ns == n

    def test_left_weight_jobv_i_continuous(self):
        """
        Test left weighting with JOBV='I' (op(V)=inv(V)).

        V must have only antistable zeros for JOBV='I'.
        Random seed: 789 (for reproducibility)
        """
        from ctrlsys import ab09jd

        np.random.seed(789)
        n, m, p = 3, 1, 1
        nv, nw = 1, 0

        a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')
        b = np.array([[1.0], [1.0], [1.0]], order='F', dtype=float)
        c = np.array([[1.0, 1.0, 1.0]], order='F', dtype=float)
        d = np.array([[0.0]], order='F', dtype=float)

        av = np.array([[1.0]], order='F', dtype=float)
        bv = np.array([[1.0]], order='F', dtype=float)
        cv = np.array([[1.0]], order='F', dtype=float)
        dv = np.array([[1.0]], order='F', dtype=float)

        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, m), order='F', dtype=float)
        cw = np.zeros((m, 1), order='F', dtype=float)
        dw = np.zeros((m, m), order='F', dtype=float)

        nr = 2
        alpha = 0.0
        tol1 = 0.0
        tol2 = 0.0

        (a_out, b_out, c_out, d_out, av_out, bv_out, cv_out,
         aw_out, bw_out, cw_out, nr_out, ns, hsv, iwarn, info) = ab09jd(
            'I', 'N', 'I', 'C', 'N', 'F', n, nv, nw, m, p, nr, alpha,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, tol1, tol2
        )

        assert info == 0, f"Expected info=0, got {info}"
        assert ns == n


class TestAB09JDRightWeighting:
    """Tests with right frequency weighting only."""

    def test_right_weight_jobw_w_continuous(self):
        """
        Test right weighting with JOBW='W' (op(W)=W).

        W must be antistable for JOBW='W'.
        Random seed: 111 (for reproducibility)
        """
        from ctrlsys import ab09jd

        np.random.seed(111)
        n, m, p = 3, 1, 1
        nv, nw = 0, 1

        a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')
        b = np.array([[1.0], [1.0], [1.0]], order='F', dtype=float)
        c = np.array([[1.0, 1.0, 1.0]], order='F', dtype=float)
        d = np.array([[0.0]], order='F', dtype=float)

        av = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, p), order='F', dtype=float)
        cv = np.zeros((p, 1), order='F', dtype=float)
        dv = np.zeros((p, p), order='F', dtype=float)

        aw = np.array([[1.0]], order='F', dtype=float)
        bw = np.array([[1.0]], order='F', dtype=float)
        cw = np.array([[1.0]], order='F', dtype=float)
        dw = np.array([[1.0]], order='F', dtype=float)

        nr = 2
        alpha = 0.0
        tol1 = 0.0
        tol2 = 0.0

        (a_out, b_out, c_out, d_out, av_out, bv_out, cv_out,
         aw_out, bw_out, cw_out, nr_out, ns, hsv, iwarn, info) = ab09jd(
            'N', 'W', 'I', 'C', 'N', 'F', n, nv, nw, m, p, nr, alpha,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, tol1, tol2
        )

        assert info == 0, f"Expected info=0, got {info}"
        assert ns == n


class TestAB09JDBothWeighting:
    """Tests with both left and right frequency weighting."""

    def test_both_weights_v_w_continuous(self):
        """
        Test both left and right weighting with JOBV='V', JOBW='W'.

        Random seed: 222 (for reproducibility)
        """
        from ctrlsys import ab09jd

        np.random.seed(222)
        n, m, p = 3, 1, 1
        nv, nw = 1, 1

        a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')
        b = np.array([[1.0], [1.0], [1.0]], order='F', dtype=float)
        c = np.array([[1.0, 1.0, 1.0]], order='F', dtype=float)
        d = np.array([[0.0]], order='F', dtype=float)

        av = np.array([[1.0]], order='F', dtype=float)
        bv = np.array([[1.0]], order='F', dtype=float)
        cv = np.array([[1.0]], order='F', dtype=float)
        dv = np.array([[1.0]], order='F', dtype=float)

        aw = np.array([[2.0]], order='F', dtype=float)
        bw = np.array([[1.0]], order='F', dtype=float)
        cw = np.array([[1.0]], order='F', dtype=float)
        dw = np.array([[1.0]], order='F', dtype=float)

        nr = 2
        alpha = 0.0
        tol1 = 0.0
        tol2 = 0.0

        (a_out, b_out, c_out, d_out, av_out, bv_out, cv_out,
         aw_out, bw_out, cw_out, nr_out, ns, hsv, iwarn, info) = ab09jd(
            'V', 'W', 'I', 'C', 'N', 'F', n, nv, nw, m, p, nr, alpha,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, tol1, tol2
        )

        assert info == 0, f"Expected info=0, got {info}"
        assert ns == n


class TestAB09JDDiscreteTime:
    """Tests for discrete-time systems."""

    def test_discrete_no_weight(self):
        """
        Test discrete-time system with no frequency weighting.

        Random seed: 333 (for reproducibility)
        """
        from ctrlsys import ab09jd

        np.random.seed(333)
        n, m, p = 3, 1, 1
        nv, nw = 0, 0

        a = np.diag([0.3, 0.5, 0.7]).astype(float, order='F')
        b = np.array([[1.0], [1.0], [1.0]], order='F', dtype=float)
        c = np.array([[1.0, 1.0, 1.0]], order='F', dtype=float)
        d = np.array([[0.0]], order='F', dtype=float)

        av = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, p), order='F', dtype=float)
        cv = np.zeros((p, 1), order='F', dtype=float)
        dv = np.zeros((p, p), order='F', dtype=float)

        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, m), order='F', dtype=float)
        cw = np.zeros((m, 1), order='F', dtype=float)
        dw = np.zeros((m, m), order='F', dtype=float)

        nr = 2
        alpha = 1.0
        tol1 = 0.0
        tol2 = 0.0

        (a_out, b_out, c_out, d_out, av_out, bv_out, cv_out,
         aw_out, bw_out, cw_out, nr_out, ns, hsv, iwarn, info) = ab09jd(
            'N', 'N', 'I', 'D', 'N', 'F', n, nv, nw, m, p, nr, alpha,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, tol1, tol2
        )

        assert info == 0, f"Expected info=0, got {info}"
        assert ns == n


class TestAB09JDJobinvModes:
    """Tests for different JOBINV modes."""

    def test_jobinv_n_inverse_free(self):
        """
        Test JOBINV='N' (inverse-free descriptor system approach).

        Random seed: 444 (for reproducibility)
        """
        from ctrlsys import ab09jd

        np.random.seed(444)
        n, m, p = 3, 1, 1
        nv, nw = 1, 0

        a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')
        b = np.array([[1.0], [1.0], [1.0]], order='F', dtype=float)
        c = np.array([[1.0, 1.0, 1.0]], order='F', dtype=float)
        d = np.array([[0.0]], order='F', dtype=float)

        av = np.array([[1.0]], order='F', dtype=float)
        bv = np.array([[1.0]], order='F', dtype=float)
        cv = np.array([[1.0]], order='F', dtype=float)
        dv = np.array([[0.5]], order='F', dtype=float)

        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, m), order='F', dtype=float)
        cw = np.zeros((m, 1), order='F', dtype=float)
        dw = np.zeros((m, m), order='F', dtype=float)

        nr = 2
        alpha = 0.0
        tol1 = 0.0
        tol2 = 0.0

        (a_out, b_out, c_out, d_out, av_out, bv_out, cv_out,
         aw_out, bw_out, cw_out, nr_out, ns, hsv, iwarn, info) = ab09jd(
            'V', 'N', 'N', 'C', 'N', 'F', n, nv, nw, m, p, nr, alpha,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, tol1, tol2
        )

        assert info == 0, f"Expected info=0, got {info}"

    def test_jobinv_a_auto_switch(self):
        """
        Test JOBINV='A' (auto-switch to inverse-free if ill-conditioned).

        Random seed: 555 (for reproducibility)
        """
        from ctrlsys import ab09jd

        np.random.seed(555)
        n, m, p = 3, 1, 1
        nv, nw = 1, 0

        a = np.diag([-1.0, -2.0, -3.0]).astype(float, order='F')
        b = np.array([[1.0], [1.0], [1.0]], order='F', dtype=float)
        c = np.array([[1.0, 1.0, 1.0]], order='F', dtype=float)
        d = np.array([[0.0]], order='F', dtype=float)

        av = np.array([[1.0]], order='F', dtype=float)
        bv = np.array([[1.0]], order='F', dtype=float)
        cv = np.array([[1.0]], order='F', dtype=float)
        dv = np.array([[1.0]], order='F', dtype=float)

        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, m), order='F', dtype=float)
        cw = np.zeros((m, 1), order='F', dtype=float)
        dw = np.zeros((m, m), order='F', dtype=float)

        nr = 2
        alpha = 0.0
        tol1 = 0.0
        tol2 = 0.0

        (a_out, b_out, c_out, d_out, av_out, bv_out, cv_out,
         aw_out, bw_out, cw_out, nr_out, ns, hsv, iwarn, info) = ab09jd(
            'V', 'N', 'A', 'C', 'N', 'F', n, nv, nw, m, p, nr, alpha,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, tol1, tol2
        )

        assert info == 0, f"Expected info=0, got {info}"


class TestAB09JDMathematicalProperties:
    """Tests for mathematical properties."""

    def test_hsv_decreasing_order(self):
        """
        Validate Hankel singular values are in decreasing order.

        Random seed: 666 (for reproducibility)
        """
        from ctrlsys import ab09jd

        np.random.seed(666)
        n, m, p = 5, 1, 1
        nv, nw = 0, 0

        a = np.diag([-1.0, -2.0, -3.0, -4.0, -5.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        av = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, p), order='F', dtype=float)
        cv = np.zeros((p, 1), order='F', dtype=float)
        dv = np.zeros((p, p), order='F', dtype=float)

        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, m), order='F', dtype=float)
        cw = np.zeros((m, 1), order='F', dtype=float)
        dw = np.zeros((m, m), order='F', dtype=float)

        nr = 0
        alpha = 0.0
        tol1 = 1e-10
        tol2 = 0.0

        (a_out, b_out, c_out, d_out, av_out, bv_out, cv_out,
         aw_out, bw_out, cw_out, nr_out, ns, hsv, iwarn, info) = ab09jd(
            'N', 'N', 'I', 'C', 'N', 'A', n, nv, nw, m, p, nr, alpha,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, tol1, tol2
        )

        assert info == 0
        for i in range(ns - 1):
            assert hsv[i] >= hsv[i + 1], f"HSV not decreasing at index {i}"

    def test_reduced_system_stability(self):
        """
        Validate reduced system preserves stability (continuous-time).

        The eigenvalues of the reduced system should have negative real parts.
        Random seed: 777 (for reproducibility)
        """
        from ctrlsys import ab09jd

        np.random.seed(777)
        n, m, p = 4, 1, 1
        nv, nw = 0, 0

        a = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        av = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, p), order='F', dtype=float)
        cv = np.zeros((p, 1), order='F', dtype=float)
        dv = np.zeros((p, p), order='F', dtype=float)

        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, m), order='F', dtype=float)
        cw = np.zeros((m, 1), order='F', dtype=float)
        dw = np.zeros((m, m), order='F', dtype=float)

        nr = 2
        alpha = 0.0
        tol1 = 0.0
        tol2 = 0.0

        (a_out, b_out, c_out, d_out, av_out, bv_out, cv_out,
         aw_out, bw_out, cw_out, nr_out, ns, hsv, iwarn, info) = ab09jd(
            'N', 'N', 'I', 'C', 'N', 'F', n, nv, nw, m, p, nr, alpha,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, tol1, tol2
        )

        assert info == 0

        if nr_out > 0:
            ar = a_out[:nr_out, :nr_out].copy()
            eigs = np.linalg.eigvals(ar)
            for eig in eigs:
                assert eig.real < 0, f"Eigenvalue {eig} has non-negative real part"


class TestAB09JDEdgeCases:
    """Edge case tests."""

    def test_n_zero(self):
        """Test with N=0 (quick return)."""
        from ctrlsys import ab09jd

        n, m, p = 0, 1, 1
        nv, nw = 0, 0

        a = np.zeros((1, 1), order='F', dtype=float)
        b = np.zeros((1, m), order='F', dtype=float)
        c = np.zeros((p, 1), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        av = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, p), order='F', dtype=float)
        cv = np.zeros((p, 1), order='F', dtype=float)
        dv = np.zeros((p, p), order='F', dtype=float)

        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, m), order='F', dtype=float)
        cw = np.zeros((m, 1), order='F', dtype=float)
        dw = np.zeros((m, m), order='F', dtype=float)

        nr = 0
        alpha = 0.0
        tol1 = 0.0
        tol2 = 0.0

        (a_out, b_out, c_out, d_out, av_out, bv_out, cv_out,
         aw_out, bw_out, cw_out, nr_out, ns, hsv, iwarn, info) = ab09jd(
            'N', 'N', 'I', 'C', 'N', 'F', n, nv, nw, m, p, nr, alpha,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, tol1, tol2
        )

        assert info == 0
        assert nr_out == 0
        assert ns == 0

    def test_m_zero(self):
        """Test with M=0 (quick return)."""
        from ctrlsys import ab09jd

        n, m, p = 2, 0, 1
        nv, nw = 0, 0

        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.zeros((n, 1), order='F', dtype=float)
        c = np.zeros((p, n), order='F', dtype=float)
        d = np.zeros((p, 1), order='F', dtype=float)

        av = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, 1), order='F', dtype=float)
        cv = np.zeros((1, 1), order='F', dtype=float)
        dv = np.zeros((1, 1), order='F', dtype=float)

        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, 1), order='F', dtype=float)
        cw = np.zeros((1, 1), order='F', dtype=float)
        dw = np.zeros((1, 1), order='F', dtype=float)

        nr = 0
        alpha = 0.0
        tol1 = 0.0
        tol2 = 0.0

        (a_out, b_out, c_out, d_out, av_out, bv_out, cv_out,
         aw_out, bw_out, cw_out, nr_out, ns, hsv, iwarn, info) = ab09jd(
            'N', 'N', 'I', 'C', 'N', 'F', n, nv, nw, m, p, nr, alpha,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, tol1, tol2
        )

        assert info == 0
        assert nr_out == 0
        assert ns == 0


class TestAB09JDErrors:
    """Error handling tests."""

    def test_invalid_jobv(self):
        """Test error for invalid JOBV parameter."""
        from ctrlsys import ab09jd

        n, m, p = 2, 1, 1

        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        av = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, p), order='F', dtype=float)
        cv = np.zeros((p, 1), order='F', dtype=float)
        dv = np.zeros((p, p), order='F', dtype=float)

        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, m), order='F', dtype=float)
        cw = np.zeros((m, 1), order='F', dtype=float)
        dw = np.zeros((m, m), order='F', dtype=float)

        with pytest.raises(ValueError):
            ab09jd(
                'X', 'N', 'I', 'C', 'N', 'F', n, 0, 0, m, p, 1, 0.0,
                a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, 0.0, 0.0
            )

    def test_invalid_dico(self):
        """Test error for invalid DICO parameter."""
        from ctrlsys import ab09jd

        n, m, p = 2, 1, 1

        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        av = np.zeros((1, 1), order='F', dtype=float)
        bv = np.zeros((1, p), order='F', dtype=float)
        cv = np.zeros((p, 1), order='F', dtype=float)
        dv = np.zeros((p, p), order='F', dtype=float)

        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, m), order='F', dtype=float)
        cw = np.zeros((m, 1), order='F', dtype=float)
        dw = np.zeros((m, m), order='F', dtype=float)

        with pytest.raises(ValueError):
            ab09jd(
                'N', 'N', 'I', 'X', 'N', 'F', n, 0, 0, m, p, 1, 0.0,
                a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, 0.0, 0.0
            )

    def test_info_18_v_not_antistable(self):
        """
        Test INFO=18: op(V) is not antistable when JOBV='V'.

        For JOBV='V', V must be antistable (all eigenvalues unstable).
        """
        from ctrlsys import ab09jd

        n, m, p = 2, 1, 1
        nv = 2

        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        av = np.array([
            [-1.0, 0.0],
            [0.0, 2.0]
        ], order='F', dtype=float)

        bv = np.eye(nv, p, order='F', dtype=float)
        cv = np.eye(p, nv, order='F', dtype=float)
        dv = np.eye(p, order='F', dtype=float)

        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, m), order='F', dtype=float)
        cw = np.zeros((m, 1), order='F', dtype=float)
        dw = np.zeros((m, m), order='F', dtype=float)

        nr = 1
        alpha = 0.0
        tol1 = 0.0
        tol2 = 0.0

        (a_out, b_out, c_out, d_out, av_out, bv_out, cv_out,
         aw_out, bw_out, cw_out, nr_out, ns, hsv, iwarn, info) = ab09jd(
            'V', 'N', 'I', 'C', 'N', 'F', n, nv, 0, m, p, nr, alpha,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, tol1, tol2
        )

        assert info == 18, f"Expected info=18, got {info}"

    def test_info_20_v_not_invertible(self):
        """Test INFO=20: V is not invertible (DV singular and NV=0)."""
        from ctrlsys import ab09jd

        n, m, p = 2, 1, 1
        nv = 0

        a = np.diag([-1.0, -2.0]).astype(float, order='F')
        b = np.ones((n, m), order='F', dtype=float)
        c = np.ones((p, n), order='F', dtype=float)
        d = np.zeros((p, m), order='F', dtype=float)

        av = np.array([[0.0]], order='F', dtype=float)
        bv = np.array([[1.0]], order='F', dtype=float)
        cv = np.array([[1.0]], order='F', dtype=float)
        dv = np.array([[0.0]], order='F', dtype=float)

        aw = np.zeros((1, 1), order='F', dtype=float)
        bw = np.zeros((1, m), order='F', dtype=float)
        cw = np.zeros((m, 1), order='F', dtype=float)
        dw = np.zeros((m, m), order='F', dtype=float)

        nr = 1
        alpha = 0.0
        tol1 = 0.0
        tol2 = 0.0

        (a_out, b_out, c_out, d_out, av_out, bv_out, cv_out,
         aw_out, bw_out, cw_out, nr_out, ns, hsv, iwarn, info) = ab09jd(
            'V', 'N', 'I', 'C', 'N', 'F', n, nv, 0, m, p, nr, alpha,
            a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, tol1, tol2
        )

        assert info == 20, f"Expected info=20, got {info}"

    def test_negative_n(self):
        """Test error for negative N."""
        from ctrlsys import ab09jd

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
            ab09jd(
                'N', 'N', 'I', 'C', 'N', 'F', -1, 0, 0, 1, 1, 0, 0.0,
                a, b, c, d, av, bv, cv, dv, aw, bw, cw, dw, 0.0, 0.0
            )
